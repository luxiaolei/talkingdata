import os
import operator
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from itertools import combinations
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)

from tensorflow.contrib.learn.python.learn.estimators import run_config
config_  = run_config.RunConfig(num_cores=12,
               save_summary_steps=40000,
               save_checkpoints_secs=18000,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000)#, allow_soft_placement= True)
 

flags = tf.app.flags

flags.DEFINE_string('model_path', '.', 'model path, default to current directory')
flags.DEFINE_string('ckpt_path', 'Sorted_lr_1e4_ed_30_layer3_simpleCross', 'model path, default to current directory')

flags.DEFINE_string('train_path', 'train_sorted.csv', 'trainning csv path')
flags.DEFINE_string('eval_path', 'eval.csv', 'evaluate csv path')
flags.DEFINE_string('test_path', 'test.csv', 'test csv path')
flags.DEFINE_string('submission_path', 'submission_model.csv', 'submission path')
flags.DEFINE_string('model_type', 'wideNdeep', 'type of model')
flags.DEFINE_integer('batch_size', 4000, 'batch size for each iter')
flags.DEFINE_integer('embed_dim', 30, 'embedding dimension for dnn part')
flags.DEFINE_integer('eval_steps', 300000, 'eval steps for each iter')
flags.DEFINE_integer('last_steps', 821667, 'last global steps')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_float('decay_rate', 0.85, 'decay rate')
flags.DEFINE_float('dropout', 0.1, 'dropout percentage')
flags.DEFINE_integer('decay_steps', 1e6, 'decay steps')
flags.DEFINE_integer('eval_mode', 24, 'mod number for eval')

flags.DEFINE_boolean('training', True, 'if False, enter testing mode')
flags.DEFINE_boolean('train_eval_set', True, 'if True, then keep trainning on the evaluation set, before prediction')

FLAGS = flags.FLAGS

dnn_hidden_units = [1000, 800, 500] # 'FLAT_embed_30_lr_5e4_layer_3', 'embed_30_lr_1e3_bs_4k_layer3'
#dnn_hidden_units = [1000, 800, 500, 100, 50] # 'embed_35_lr_5e4_bs_1e4_layer_5'


LABEL_COLUMN = 'group_mapped'

CATEGORICAL_COLUMNS = ['5_1_day', '5_2_day', '5_3_day', '5_4_day', '5_5_day', '5_6_day',
       '5_7_day', 'H0_to_2', 'H0_to_6', 'H0_to_9', 'H12_to_14',
       'H14_to_18', 'H18_to_20', 'H18_to_23', 'H19_to_23', 'H1_to_5',
       'H20_to_22', 'H2_to_6', 'H4_to_8', 'H6_to_9', 'H8_to_10',
       'H9_to_12', 'H9_to_14', 'H9_to_18', 'app_active_count', 'app_count',
       'avg_events_perday', 'event_count', 'geo_20_1_count',
       'geo_20_2_count', 'geo_20_3_count', 'geo_35_1_count',
       'geo_35_2_count', 'geo_35_3_count', 'geo_count_51holiday',
       'geo_count_None_51holiday',  'holiday_51_count',
       'hour_dt_count', 'hour_nt_count', 'weekday_count', 'weekend_count',
       '1_free', 'Cards_RPG', 'Casual_puzzle_categories', 'Custom_label',
       'Industry_tag', 'Irritation_/_Fun_1', 'Personal_Effectiveness_1',
       'Property_Industry_2.0', 'Property_Industry_new', 'Relatives_1',
       'Services_1', 'Tencent', 'game', 'unknown', 
        'dt_count_51holiday', 'phone_brand', 'device_model',
       'dt_count_None_51holiday', 'nt_count_51holiday',
       'nt_count_None_51holiday',
       'geo_label_20', 'geo_label_35', 'day_of_week', 'hour',
       'date', 'is_installed', 'is_active', 'TS_1_free',
       'TS_Cards_RPG', 'TS_Casual_puzzle_categories', 'TS_Custom_label',
       'TS_Industry_tag', 'TS_Irritation_/_Fun_1',
       'TS_Personal_Effectiveness_1', 'TS_Property_Industry_2.0',
       'TS_Property_Industry_new', 'TS_Relatives_1', 'TS_Services_1',
       'TS_Tencent', 'TS_game', 'TS_unknown']

complex_cols = ['phone_brand', 'device_model', 'H9_to_18', '5_4_day',
				'weekend_count', 'weekday_count', 'H0_to_9', '5_1_day',
				'geo_35_1_count', 'dt_count_None_51holiday', 'nt_count_None_51holiday',
				'date', 'geo_label_20', 'TS_Irritation_/_Fun_1', 
				'TS_Personal_Effectiveness_1', 'TS_Property_Industry_2.0',
				'TS_Services_1', 'is_active', 'hour']

simple_cols = ['phone_brand', 'device_model', 'H9_to_18', '5_4_day',
				'H0_to_9', '5_1_day', 
				'date', 'geo_label_20', 'hour']
CROSS_LIST_complex = list(combinations(complex_cols,2)) + list(combinations(complex_cols,3))

CROSS_LIST_simple = list(combinations(simple_cols,2)) + list(combinations(simple_cols,3))
print('Currently, we are using simpler crossed columns:')
print(len(CROSS_LIST_simple))
print('*'*100)

def papare_data(i=0):
	type_dic = {cate_col: np.str for cate_col in CATEGORICAL_COLUMNS}
	type_dic.update({'device_id': np.str, 'group': np.str})

	train = pd.read_csv(os.path.join(FLAGS.model_path, FLAGS.train_path), dtype=type_dic)
	eval = pd.read_csv(os.path.join(FLAGS.model_path, FLAGS.eval_path), dtype=type_dic)
	if FLAGS.training:
		test = 0
	else:                                               
		test = pd.read_csv(os.path.join(FLAGS.model_path, FLAGS.test_path), dtype=type_dic)

    
	sub_template = pd.read_csv('../csvs/sample_submission.csv', dtype={'device_id':np.str})

	# transform labels
	label_dict = {i:k for k,i in enumerate(train['group'].unique().tolist())}
	train['group_mapped'] = train['group'].apply(lambda x: label_dict[x])
	eval['group_mapped'] = eval['group'].apply(lambda x: label_dict[x])

	# dummy code
	class_cont = train[LABEL_COLUMN].unique().shape[0]
	return train, test, eval, sub_template, label_dict, class_cont




def _cross_columns_2d(cross_base_cols1, cross_base_cols2, cross_base_columns):
	crossed_columns = []
	for c1 in cross_base_cols1:
		c1_layer = next((layer for layer in cross_base_columns
			if layer.name == c1 or layer.name == c1+'_BUCKETIZED'), None)
		for c2 in cross_base_cols2:
			c2_layer = next((layer for layer in cross_base_columns
			if layer.name == c2  or layer.name == c2+'_BUCKETIZED' ), None)
			crossed_columns +=[tf.contrib.layers.crossed_column([c1_layer, c2_layer], hash_bucket_size=int(1e5))] #flaw
	return crossed_columns

def _cross_columns(col_names_list, cross_base_columns):
	layers = [layer for layer in cross_base_columns 
				if layer.name in col_names_list]
	#assert len(col_names_list) == len(layers)

	return [tf.contrib.layers.crossed_column(layers, hash_bucket_size=int(1e2))]


def model(class_cont=12, dnn_hidden_units= [300, 500, 100, 50]):
	# Contruct base columns
	categorical_base_cols = [tf.contrib.layers.sparse_column_with_hash_bucket(column_name=name, hash_bucket_size=10) for name in CATEGORICAL_COLUMNS]
	categorical_embed_base_cols = [tf.contrib.layers.embedding_column(cate_layer, dimension=FLAGS.embed_dim) for cate_layer in categorical_base_cols]


	# Construct crossed Columns
	crossed_columns, crossed_columns_2d = [], []
	cross_base_columns = categorical_base_cols

	for col_list in CROSS_LIST_simple:
		crossed_columns += _cross_columns(col_list, cross_base_columns)
	assert isinstance(categorical_base_cols, list), isinstance(crossed_columns, list)
	# Construct wide and deep columns
	wide_columns = categorical_base_cols + crossed_columns
	deep_columns = categorical_embed_base_cols


	# create model
    # setup exponential decay function
	def optimizer_exp_decay():
	  global_step = tf.contrib.framework.get_or_create_global_step()
	  learning_rate = tf.train.exponential_decay(
	      learning_rate=FLAGS.learning_rate, global_step=global_step,
	      decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate, )
	  return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    
	linear_optimizer=tf.train.FtrlOptimizer(learning_rate=FLAGS.learning_rate ,learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.2, l2_regularization_strength=0.3)
	#with tf.device('/gpu:2'):


 	if FLAGS.model_type == "wide":
 		m = tf.contrib.learn.LinearClassifier(model_dir=FLAGS.ckpt_path,
                                          feature_columns=wide_columns,
                                          n_classes=class_cont,
                                          config= config_)
 	elif FLAGS.model_type == "deep":
 		m = tf.contrib.learn.DNNClassifier(model_dir=FLAGS.ckpt_path,
                                       feature_columns=deep_columns,
                                       hidden_units=dnn_hidden_units,
                                       n_classes=class_cont,
                                       config= config_,
                                       optimizer=optimizer_exp_decay,
                                       dropout = FLAGS.dropout)
 	else:
		m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=FLAGS.ckpt_path, n_classes=class_cont, 
                                                 linear_feature_columns=wide_columns, 
                                                 dnn_feature_columns=deep_columns,
                                                 dnn_hidden_units= dnn_hidden_units,
                                                 config= config_,
                                                dnn_optimizer=optimizer_exp_decay,
                                                    dnn_dropout = FLAGS.dropout)
                                                   # dnn_activation_fn = tf.nn.relu6)
	return m

 

# pepare data function
def input_fn(df, train=True):
	categorical_cols = {k: tf.SparseTensor(indices=[[i,0] for i in range(df[k].size)], values=df[k].values, shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
	feature_cols= dict(categorical_cols)
	if train:
		label = tf.constant(df[LABEL_COLUMN].values)
		return feature_cols, label
	else:
		return feature_cols



def train_model(m, df_train, start, stop):
	df_t = df_train.iloc[start :stop, :]
	print('df_t shape: {0}'.format(df_t.shape))
	m.fit(input_fn=lambda: input_fn(df_t), steps=FLAGS.batch_size)#, save_steps=10000)


def test_model(m, df):
	# get results
	ans = m.predict_proba(input_fn=lambda: input_fn(df, train=False))
	ans = np.round(ans, decimals=4)
	return ans


def main(_):
	df_train, df_test, df_eval, sub_template, label_dict, class_cont =  papare_data()

	# train the model
	m = model(dnn_hidden_units= dnn_hidden_units)
	if FLAGS.training:
		last_trained_steps = FLAGS.last_steps
		max_step = df_train.shape[0] - last_trained_steps
		for i in range(max_step/FLAGS.batch_size):
			start = last_trained_steps+(i*FLAGS.batch_size)
			stop = start + FLAGS.batch_size
			train_model(m, df_train, start, stop)
			# evaluate on trained and df_eval data, with randomly select 200 instances
			if i % FLAGS.eval_mode == 0:
				train_indx, eval_indx = np.random.randint(0, stop, FLAGS.eval_steps), np.random.randint(0, df_eval.shape[0], FLAGS.eval_steps)
				df_eval_tr = df_train.iloc[train_indx].drop_duplicates(['device_id'])
				df_eval_el = df_eval.iloc[eval_indx].drop_duplicates(['device_id'])
				train_result = m.evaluate(input_fn=lambda: input_fn(df_eval_tr), steps = 1, name='train')
				eval_result = m.evaluate(input_fn=lambda: input_fn(df_eval_el), steps = 1, name='valid')
				print('Trainning result after {0} steps:'.format(stop))
				print(train_result)
				print('Evaluation result:')
				print(eval_result)
	else:
		if FLAGS.train_eval_set:  
			m.fit(input_fn=lambda: input_fn(df_eval), steps=df_eval.shape[0])
		# do evaluation on test set, and create submission csv
		total_steps = df_test.shape[0]
		eval_steps = FLAGS.batch_size
		ans_array = np.empty((0,12))
		for num_runs in range(int(total_steps/eval_steps)):
		    start = num_runs*eval_steps
		    stop = (num_runs+1)*eval_steps
		    test_batch = df_test.iloc[start:stop, :]
		    ans = test_model(m, test_batch)
		    ans_array = np.vstack((ans_array, ans))
            #print('***************************************')
		    print '***************************************', start,stop, '***************************************'

		test_batch_last = df_test.iloc[ans_array.shape[0]:, :]
		ans = test_model(m, test_batch_last)
		ans_array = np.vstack((ans_array, ans))

		# construct dataframe
		sorted_col_tups = sorted(label_dict.items(), key=operator.itemgetter(1))
		sorted_cols = [t[0] for t in sorted_col_tups]
		df_ans = pd.DataFrame(ans_array, columns=sorted_cols)
		df_ans['device_id'] = df_test['device_id']

		# groupby id, and agg mean
		df_ans_gped = df_ans.groupby('device_id').agg(np.mean).round(4)

		df_ans_gped.to_csv('tem_results.csv')
        # reorder the columns and create file
		df_ans_gped['device_id'] = df_ans_gped.index.values
		df_ans_gped = df_ans_gped[sub_template.columns]
		sub_template['temp'] = sub_template.index.values
		sub_template = sub_template[['device_id', 'temp']]
		sub_df = pd.merge(sub_template, df_ans_gped, on='device_id', how='left')
		sub_df.drop('temp', axis=1, inplace=True)
		sub_df.to_csv(FLAGS.submission_path,index=False)

if __name__ == '__main__':
	tf.app.run()





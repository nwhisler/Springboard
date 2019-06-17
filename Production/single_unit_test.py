import os
import pandas as pd
import numpy as np
from unittest import TestCase
from twitter import twitter_model
from warnings import filterwarnings

class Twitter_Test(TestCase):

	def setUp(self):
        
		filterwarnings('ignore','elementwise')
		filterwarnings('ignore','unclosed')
		filterwarnings('ignore','numpy')
		df_data = pd.read_csv('testing_data.csv')
		data = df_data[df_data.columns[1:]].to_json()
		self.model = twitter_model(data)

	def test_model(self):

		# Test sample input.
		self.assertTrue(len(self.model.df)>0)

		# Test dataframe columns.
		for col in self.model.columns_required:

			self.assertIn(col,self.model.current_columns)
			self.assertEqual(len(self.model.df[col][self.model.df[col].isna()]),0)         
			self.assertEqual(self.model.df[col].dtype,np.dtype(self.model.column_dtypes[col]))

		# Test text type.
		for sample in self.model.df.text:

			self.assertIsInstance(sample,str)
 
		# Test embedded vec file path.
		os.path.exists('../glove.6B.300d.txt')

		# Test training embedded vec file path, sample count, and shape.
		os.path.exists('embedded_vecs/train_embedded_vecs.csv')

		self.assertTrue(self.model.train_embedded_vecs.shape[0]>0)	

		self.assertEqual(self.model.train_embedded_vecs.shape[1],300)

		# Test training norm vec file path and sample count.
		os.path.exists('embedded_vec_norms/train_norms.csv')

		self.assertTrue(self.model.train_norms.shape[0]>0)

		# Test training entity list.
		self.assertTrue(len(self.model.entity_list)>0)

		# Test training pos tag list.
		self.assertTrue(len(self.model.pos_tag_list)>0)

		# Test text_tokens function.
		self.model.text_tokens()

		self.assertTrue(len(self.model.df.tokens)>0)
		self.samples = len(self.model.df.tokens)

		# Test sentiment function.
		self.model.sentiment()

		self.assertTrue(len(self.model.df.sentiment)>0)
		self.assertEqual(self.samples,len(self.model.df.sentiment))

		# Test pos function.
		self.model.pos()

		self.assertEqual(list(self.model.df_pos.columns),self.model.pos_tag_list)
		self.assertEqual(self.samples,len(self.model.df_pos))

		# Test entity function.
		self.model.entity()

		self.assertEqual(list(self.model.df_entities.columns),self.model.entity_list)
		self.assertEqual(self.samples,len(self.model.df_entities))

		# Test embed function.
		self.model.embed_vecs()

		self.assertEqual(self.model.embedded_vecs.shape[0],self.samples)
		self.assertEqual(self.model.embedded_vecs.shape[1],300)

		# Test norm function.
		self.model.tweet_norms()

		self.assertEqual(self.model.norm_vecs.shape[0],self.samples)
		self.assertEqual(len(self.model.norm_vecs[np.isinf(self.model.norm_vecs)]),0)
		self.assertEqual(len(self.model.norm_vecs[np.isnan(self.model.norm_vecs)]),0)

		# Test cos_distance function.
		self.model.embedded_cos_distance()

		tweets = self.model.df.cos_vecs.values
		self.assertEqual(tweets.shape[0],self.samples)
		self.assertEqual(len(tweets[np.isinf(tweets)]),0)
		self.assertEqual(len(tweets[np.isnan(tweets)]),0)

		 # Test vec_diff function.
		self.model.vec_diff()

		self.assertEqual(self.model.df.delta_vecs.shape[0],self.samples)

		# Test evaluate function.
		os.path.exists('boost_model.bin')
		predictions = self.model.evaluate()
		self.assertEqual(len(predictions),self.samples)
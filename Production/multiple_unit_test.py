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

	def test_df(self):

		self.assertTrue(len(self.model.df)>0)

	def test_columns(self):

		for col in self.model.columns_required:

			self.assertIn(col,self.model.current_columns)

	def test_text(self):

		for sample in self.model.df.text:

			self.assertIsInstance(sample,str)
            
	def test_na_values(self):
    
		for col in self.model.columns_required:

			self.assertEqual(len(self.model.df[col][self.model.df[col].isna()]),0)         

	def test_column_dtypes(self):
    
		for col in self.model.columns_required:
              
			self.assertEqual(self.model.df[col].dtype,np.dtype(self.model.column_dtypes[col]))            
            
	def test_path(self):

			os.path.exists('../glove.6B.300d.txt')

	def test_train_embedded_vecs(self):

		os.path.exists('embedded_vecs/train_embedded_vecs.csv')

		self.assertTrue(self.model.train_embedded_vecs.shape[0]>0)	

		self.assertEqual(self.model.train_embedded_vecs.shape[1],300)

	def test_train_norms(self):

		os.path.exists('embedded_vec_norms/train_norms.csv')

		self.assertTrue(self.model.train_norms.shape[0]>0)

	def test_entity_list(self):

		self.assertTrue(len(self.model.entity_list)>0)

	def test_pos_tag_list(self):

		self.assertTrue(len(self.model.pos_tag_list)>0)

	def test_tokens(self):

		self.model.text_tokens()

		self.assertTrue(len(self.model.df.tokens)>0)
		self.samples = len(self.model.df.tokens)

	def test_sentiment(self):

		self.model.text_tokens()
		self.model.sentiment()

		self.assertTrue(len(self.model.df.sentiment)>0)
		self.assertEqual(len(self.model.tokens),len(self.model.df.sentiment))

	def test_pos(self):

		self.model.text_tokens()
		self.model.pos()

		self.assertEqual(list(self.model.df_pos.columns),self.model.pos_tag_list)
		self.assertEqual(len(self.model.tokens),len(self.model.df_pos))

	def test_entity(self):

		self.model.text_tokens()
		self.model.pos()
		self.model.entity()

		self.assertEqual(list(self.model.df_entities.columns),self.model.entity_list)
		self.assertEqual(len(self.model.tokens),len(self.model.df_entities))

	def test_embed_vecs(self):

		self.model.text_tokens()
		self.model.embed_vecs()

		self.assertEqual(self.model.embedded_vecs.shape[0],len(self.model.tokens))
		self.assertEqual(self.model.embedded_vecs.shape[1],300)

	def test_norms(self):

		self.model.text_tokens()
		self.model.embed_vecs()
		self.model.tweet_norms()

		self.assertEqual(self.model.norm_vecs.shape[0],len(self.model.tokens))
		self.assertEqual(len(self.model.norm_vecs[np.isinf(self.model.norm_vecs)]),0)
		self.assertEqual(len(self.model.norm_vecs[np.isnan(self.model.norm_vecs)]),0)

	def test_cos_vecs(self):

		self.model.text_tokens()
		self.model.embed_vecs()
		self.model.tweet_norms()
		self.model.embedded_cos_distance()

		tweets = self.model.df.cos_vecs.values
		self.assertEqual(tweets.shape[0],len(self.model.tokens))
		self.assertEqual(len(tweets[np.isinf(tweets)]),0)
		self.assertEqual(len(tweets[np.isnan(tweets)]),0)

	def test_vec_diff(self):

		self.model.text_tokens()
		self.model.embed_vecs()
		self.model.vec_diff()

		self.assertEqual(self.model.df.delta_vecs.shape[0],len(self.model.tokens))

	def test_model(self):
        
		self.model.feature_generation()

		os.path.exists('boost_model.bin')
		predictions = self.model.evaluate()
		self.assertEqual(len(predictions),len(self.model.tokens))
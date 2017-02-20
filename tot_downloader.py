import pymongo
from pymongo import MongoClient
import json
import pandas as pd
import numpy as np

#Connect to MongoDB
client = MongoClient('mongodb://bonnyberliner2:test123@ds143539.mlab.com:43539/heroku_l96r9wz3')

#Only one database, so just get the default
db = client.get_default_database()

#Get all of the table names
collections = db.collection_names()
def get_shoe_df():
	#Get all of the items in the shoes table
	shoes = db.shoes.find()
	needed_shoe_cols = ['Binding','Brand','Color', 'Department','Feature','Label','ProductGroup','ProductTypeName', 'Publisher','Studio','Title']

	#Load SQL query output into a dictionary and then use pd.DataFrame.from_dict()
	shoeDict = {}
	for n, shoe in enumerate(shoes):
		#Setup new empty dict and grab id and ASIN
		shoeDict[n] = {}
		shoe_id = shoe['_id']
		ASIN = json.loads(shoe['product_info']['ASIN'])[0]

		shoeDict[n]['_id'] = shoe_id
		shoeDict[n]['ASIN'] = ASIN

		#populate with needed columns (only flat columns)

		#first get itemAttributes
		itemAttributes = json.loads(shoe['product_info']['ItemAttributes'])[0]

		for col in needed_shoe_cols:
			try:
				shoeDict[n][col] = itemAttributes[col][0]
			except:
				shoeDict[n][col] = np.NaN
				pass

	#turn shoe dict into a DF
	shoeDF = pd.DataFrame.from_dict(shoeDict, orient='index')
	return shoeDF

def get_vote_df():
	votes = db.votes.find()

	voteDict = {}

	for n, vote in enumerate(votes):
		voteDict[n] = vote

	voteDF = pd.DataFrame.from_dict(voteDict, orient='index')
	return voteDF





#!/usr/bin/env python

import boto3
import numpy as np
import pickle
import argparse
import ast
try:
	import ember
except:
	os.system('wget https://github.com/endgameinc/ember/archive/master.zip')
	os.system('unzip master.zip')
	os.system('rm master.zip')
	os.system('cp -r ember-master/* .')
	os.system('rm -r ember-master')
	os.system('pip install -r requirements.txt')
	os.system('python setup.py install')
	import ember
import json
with open("mms_scaler","rb") as f:
  mms = pickle.load(f)
  f.close()

def main():
	prog = "predict_sample"
	descr = "Using the model to predict a single PE's binary."
	parser = argparse.ArgumentParser(prog=prog, description=descr)
	parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
	parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
	args = parser.parse_args()

	extractor = ember.PEFeatureExtractor(args.featureversion)
	sample_data = open(args.binaries[0],'rb').read()
	sample_data = extractor.feature_vector(sample_data)
	sample_data = np.array(sample_data, dtype=np.float32)
	sample_data = mms.transform([sample_data])
	sample_data = np.reshape(sample_data,(-1,1,2381))
	sample_data = sample_data.tolist()


	client = boto3.client('runtime.sagemaker',
	region_name='us-east-1',
	aws_access_key_id='<put access key id here>',
	aws_secret_access_key='<put secret key here>',
	aws_session_token='<put token here>'
)
	response = client.invoke_endpoint(EndpointName='sagemaker-tensorflow-2020-04-12-12-12-55-002', Body=json.dumps(sample_data))
	response_body = response['Body']
	bbtyes = response_body.read()
	astr = bbtyes.decode("UTF-8")
	d = ast.literal_eval(astr)
	d = d['outputs']['score']['floatVal']

	if d[0] >=.5:
		print("Malicious")
	else:
		print("Benign")
    


if __name__ == "__main__":
    main()

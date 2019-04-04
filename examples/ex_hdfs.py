import pandas as pd
import numpy as np
import h5py
from hdfs.client import Client
from pyhdfs import HdfsClient

'''
import pyarrow as pa
pa.hdfs.connect(host='192.168.0.186',port=9870,user='yanrujing')
#pa.hdfs.connect()
'''


client = HdfsClient(hosts='192.168.0.186:9870',user_name='yanrujing')

a1=client.open('/r2/test/transformed.h5')
#a1=client.open('/r2/userData/6c0f5b62c3624f6bad70b0a3066e9085/1/csv_header.csv')
b1=a1.read()

c1=h5py.File('/home/chen/桌面/transformed.h5')
#http://192.168.0.186:9870/explorer.html#/r2/test/transformed.h5

'''
client2 = Client(url="http://192.168.0.186:9870",root='yanrujing')
# client2.read('/r2/userData/6c0f5b62c3624f6bad70b0a3066e9085/1/csv_header.csv')
# a2=client2.read('/r2/userData/6c0f5b62c3624f6bad70b0a3066e9085/1/csv_header.csv')

with client2.read('/r2/userData/6c0f5b62c3624f6bad70b0a3066e9085/1/csv_header.csv') as reader:
    #a=pd.read_csv(reader)
    content = reader.read()
'''

a1=open('/home/chen/桌面/transformed.h5','rb')
b1=h5py.File(a1,'r')
print(b1)


print(1)

from io import BytesIO

b3=BytesIO()

with h5py.File(buf) as store:
    store['a']=[1,2,3]
    store['b']=[4,5,6]


with open('/home/chen/桌面/transformed.h5') as f:

    print(f)


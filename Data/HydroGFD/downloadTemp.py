import cdsapi

dataset = "sis-ecv-cmip5-bias-corrected"
request = {
    'variable': 'mean_2m_temperature',
    'model': 'gfdl_cm3',
    'experiment': 'rcp_4_5',
    'period': ['19600101_19641231', '19650101_19691231', '19700101_19741231', '19750101_19791231', '19800101_19841231', '19850101_19891231', '19900101_19941231', '19950101_19991231', '20000101_20041231', '20050101_20051231', '20060101_20101231', '20110101_20151231', '20160101_20201231', '20210101_20251231']
}

client = cdsapi.Client()
client.retrieve(dataset, request, "datasets/temp.zip").download()

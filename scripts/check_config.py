from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ConfigCheck") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .getOrCreate()

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
iterator = hadoop_conf.iterator()
while iterator.hasNext():
    prop = iterator.next()
    if "60s" in prop.getValue():
        print(f"{prop.getKey()} = {prop.getValue()}")

spark.stop()

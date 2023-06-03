import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']




def create_spark_session():
    """
    Creates a Spark session

    Arguments:
        None
    
    Returns:
        spark: Spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark




def process_song_data(spark, input_data, output_data):
    """
    Reads the song data json-files, processes them, 
    and writes songs table and artists table to AWS S3

    Arguments:
        spark: Spark session
        input_data: Path to the input data folder
        output_data: Path to the output data folder

    Returns:
        None
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_data_output = os.path.join(output_data, 'song_table.parquet')
    songs_table.write.partitionBy('year','artist_id').parquet(songs_data_output, 'overwrite')

    # extract columns to create artists table
    artists_table = df.select(["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]).distinct()
    
    # write artists table to parquet files
    artists_data_output = os.path.join(output_data, 'artists_table.parquet')
    artists_table.write.parquet(artists_data_output, 'overwrite')

    # Create a view and cache the song DataFrame to be able to access it later
    df.createOrReplaceTempView('song_df')
    spark.table('song_df')
    spark.table('song_df').cache
    spark.table('song_df').count




def process_log_data(spark, input_data, output_data):
    """
    Reads the json-files from log data, processes them, 
    and writes users table, time table and songplays table to AWS S3

    Arguments:
        spark: Spark session
        input_data: Path to the input data folder
        output_data: Path to the output data folder

    Returns:
        None
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.select('userId', 'firstName', 'lastName', 'gender', 'level').distinct()
    
    # write users table to parquet files
    users_data_output = os.path.join(output_data, 'users_table.parquet')
    users_table.write.parquet(users_data_output, 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp_udf = udf(lambda unit: int(unit / 1000))
    df = df.withColumn('timestamp', get_timestamp_udf('ts').cast('Integer'))
    
    # create datetime column from original timestamp column
    get_datetime_udf = udf(lambda ts: str(datetime.fromtimestamp(ts)))
    df = df.withColumn('start_time', get_datetime_udf('timestamp').cast('Timestamp'))
    
    # extract columns to create time table
    time_table = df.select('start_time') \
                   .withColumn('hour', hour('start_time').cast('Integer')) \
                   .withColumn('day', dayofmonth('start_time').cast('Integer')) \
                   .withColumn('week', weekofyear('start_time').cast('Integer')) \
                   .withColumn('month', month('start_time').cast('Integer')) \
                   .withColumn('year', year('start_time').cast('Integer')) \
                   .withColumn('weekday', date_format('start_time', 'u').cast('Integer')) \
                   .distinct()
    
    # write time table to parquet files partitioned by year and month
    time_data_output = os.path.join(output_data, 'time_table.parquet')
    time_table.write.partitionBy('year','month').parquet(time_data_output, 'overwrite')

    # read in song data to use for songplays table
    song_df = spark.sql('SELECT DISTINCT song_id, title, artist_id, artist_name FROM song_df')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_data_join = df.join(song_df, (df.song==song_df.title) & (df.artist==song_df.artist_name), how='left')
    songplays_table = songplays_data_join.select(monotonically_increasing_id().alias('songplay_id'),
                                                    'start_time',
                                                    'userId',
                                                    'level',
                                                    'song_id', 
                                                    'artist_id', 
                                                    'sessionId', 
                                                    'location', 
                                                    'userAgent',
                                                    month(col("start_time")).alias("month"),
                                                    year(col("start_time")).alias("year")
                                                )

    # write songplays table to parquet files partitioned by year and month
    songplays_data_output = os.path.join(output_data, 'songplays_table.parquet', partitionBy=['year', 'month'])
    songplays_table.write.parquet(songplays_data_output, 'overwrite')





def main():
    """
    Provides the input_data and output_data paths 
    and triggers the functions process_song_data and process_log_data.
    
    Arguments:
        None
        
    Returns:
        None
    """
    spark = create_spark_session()
    input_data = config['IN']['INPUT_DATA_PATH']
    output_data = config['OUT']['OUTPUT_PATH']
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)




if __name__ == "__main__":
    main()
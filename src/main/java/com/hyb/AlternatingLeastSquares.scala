package com.hyb

import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._

import scala.util.Random

/**
  * 交替最小二乘算法ALS
 */
object AlternatingLeastSquares {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("decision tree")
      .setMaster("local[4]")
      //.set("spark.debug.maxToStringFields", "1000")
    val spark = SparkSession.builder.config(conf).getOrCreate()

    val rawUserArtistData = spark.read
      .textFile("G:\\data\\profiledata_06-May-2005\\user_artist_data.txt")
      .repartition(4)
    val rawArtistData = spark.read
      .textFile("G:\\data\\profiledata_06-May-2005\\artist_data.txt")
      .repartition(4)
    val rawArtistAlias = spark.read
      .textFile("G:\\data\\profiledata_06-May-2005\\artist_alias.txt")
      .repartition(4)

    import rawUserArtistData.sparkSession.implicits._
    val userArtistDF=rawUserArtistData.map{line=>
      val Array(user,artist,_*)=line.split(' ')
      (user.toInt,artist.toInt)
    }.toDF("user","artist")
    userArtistDF.agg(min("user"),max("user")
      ,min("artist"),max("artist")).show()

    val artistByID= rawArtistData.flatMap{line=>
      val(id,name)=line.span(_!='\t')
      if(name.isEmpty){
        None
      }else{
        try{
          Some((id.toInt,name.trim))
        }catch {
          case _:NumberFormatException=>None
        }
      }
    }.toDF("id","name")

    val artistAlias=rawArtistAlias.flatMap{line=>
      val Array(artist,alias)=line.split('\t')
      if(artist.isEmpty){
        None
      }else{
        Some((artist.toInt,alias.toInt))
      }
    }.collect().toMap

    val bArtistAlias=spark.sparkContext.broadcast(artistAlias)
    val trainData=buildCounts(rawUserArtistData,bArtistAlias)
    trainData.cache()

    val model=new ALS()
      .setSeed(Random.nextLong())
      .setImplicitPrefs(true)
      .setRank(10)
      .setRegParam(0.01)
      .setAlpha(1.0)
      .setMaxIter(5)
      .setUserCol("user")
      .setItemCol("artist")
      .setRatingCol("count")
      .setPredictionCol("prediction")
      .fit(trainData)

    //model.userFactors.show(truncate = false)

    val userID=2093760
    val existingArtistIDs=trainData
      .filter($"user"===userID)
      .select("artist").as[Int].collect()

    artistByID.filter($"id" isin(existingArtistIDs:_*)).show()

    val result= model.recommendForAllUsers(10)
    result.show()



  }
  def buildCounts(rawUserArtistData:Dataset[String],
                  bArtistAlias:Broadcast[Map[Int,Int]]):DataFrame={
    import rawUserArtistData.sparkSession.implicits._

    rawUserArtistData.map{line=>
      val Array(userID,artistID,count)= line.split(' ').map(_.toInt)
      val finalArtistID=bArtistAlias.value.getOrElse(artistID,artistID)
      (userID,finalArtistID,count)
    }.toDF("user","artist","count")
  }




}

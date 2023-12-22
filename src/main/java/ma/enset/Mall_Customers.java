package ma.enset;

import org.apache.hadoop.mapreduce.Cluster;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Mall_Customers {
    public static void main(String[] args) {
        SparkSession ss= SparkSession.builder().appName("Tp spark ml").master("local[*]").getOrCreate();
        Dataset<Row> dataset =ss.read().option("inferSchema",true).option("header",true).csv("Mall_Customers.csv");
        VectorAssembler assembler=new VectorAssembler().setInputCols(new String[]{"Age","Annual Income (k$)","Spending Score (1-100)"}
        ).setOutputCol("features");
        Dataset<Row> assembleDataset = assembler.transform(dataset);
        MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("normalizeFeatures");
        Dataset<Row> normalizeDS = scaler.fit(assembleDataset).transform(assembleDataset);
        normalizeDS.printSchema();
        KMeans kMeans = new KMeans().setK(5)
                .setSeed(123).setFeaturesCol("normalizeFeatures").setPredictionCol("prediction");
        KMeansModel model = kMeans.fit( normalizeDS ) ;
        Dataset < Row > predictions= model.transform ( normalizeDS ) ;
        predictions.show ( 200 ) ;
        ClusteringEvaluator evaluator = new ClusteringEvaluator( ) ;
        double score =evaluator.evaluate ( predictions ) ;
        System.out.println ( score);

    }
}

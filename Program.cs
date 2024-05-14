using Microsoft.ML.OnnxRuntime;
using OnnxRuntime.ResNet.Template.utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxRuntime.ResNet.Template
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Read paths
            string modelFilePath = @"D:\OJT_SU24\ONNX-Template\model\resnet50v2.onnx";
            string imageFilePath = @"D:\OJT_SU24\ONNX-Template\data\dog.jpeg";

            var input = ImageHelper.GetImageTensorFromPath(imageFilePath);
            var top10 = ModelHelper.GetPredictions(input, modelFilePath);

            // Print results to console
            Console.WriteLine("Top 10 predictions for ResNet50 v2...");
            Console.WriteLine("--------------------------------------------------------------");
            foreach (var t in top10)
            {
                Console.WriteLine($"Label: {t.Label}, Confidence: {t.Confidence}");
            }
        }
    }
}

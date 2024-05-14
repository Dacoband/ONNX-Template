using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

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
            string modelFilePath = @"D:\OJT_SU24\ONNX-Template\model\test_wb.onnx";
            var imageFilePath = new List<string>
    {
        @"D:\OJT_SU24\ONNX-Template\data\8D5U5524_D.png",
        @"D:\OJT_SU24\ONNX-Template\data\8D5U5524_S.png",
        @"D:\OJT_SU24\ONNX-Template\data\8D5U5524_T.png"
    };

            // Get tensors for each image
            var input1 = ImageHelper.GetImageTensorFromPath(imageFilePath[0]);
            var input2 = ImageHelper.GetImageTensorFromPath(imageFilePath[1]);
            var input3 = ImageHelper.GetImageTensorFromPath(imageFilePath[2]);

            var combinedInput = ConcatenateTensors(input1, input2, input3);


            ReadOnlySpan<int> shape = combinedInput.Dimensions;
            Console.WriteLine(string.Join(", ", shape.ToArray()));


            // var output = ModelHelper.GetPredictions(combinedInput, modelFilePath);

            //Console.WriteLine($"Output: {string.Join(",", output.)}");
        }

        private static Tensor<float> ConcatenateTensors(Tensor<float> tensor1, Tensor<float> tensor2, Tensor<float> tensor3)
        {
            var tensor1Shape = tensor1.Dimensions;
            var tensor2Shape = tensor2.Dimensions;
            var tensor3Shape = tensor3.Dimensions;

            var combinedTensor = new DenseTensor<float>(new[] { tensor1Shape[0], tensor1Shape[1] * 3, tensor1Shape[2], tensor1Shape[3] });

            var combinedArray = combinedTensor.ToArray();
            var tensor1Array = tensor1.ToArray();
            var tensor2Array = tensor2.ToArray();
            var tensor3Array = tensor3.ToArray();

            Array.Copy(tensor1Array, 0, combinedArray, 0, tensor1Array.Length);

            Array.Copy(tensor2Array, 0, combinedArray, tensor1Array.Length, tensor2Array.Length);

            Array.Copy(tensor3Array, 0, combinedArray, tensor1Array.Length + tensor2Array.Length, tensor3Array.Length);

            return combinedTensor;
        }


    }
}

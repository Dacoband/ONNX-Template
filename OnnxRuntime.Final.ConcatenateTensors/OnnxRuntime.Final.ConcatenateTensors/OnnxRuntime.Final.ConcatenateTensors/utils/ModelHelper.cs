using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxRuntime.Final.ConcatenateTensors.utils
{
    public static class ModelHelper
    {
        public static Tensor<float> GetPredictions(Tensor<float> input, string modelFilePath)
        {
            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("modelInput", input)
            };
            // Run inference
            var session = new InferenceSession(modelFilePath);
            var results = session.Run(inputs);
            var resultTensor = results.First().AsTensor<float>();
            return resultTensor;
        }
    }
}

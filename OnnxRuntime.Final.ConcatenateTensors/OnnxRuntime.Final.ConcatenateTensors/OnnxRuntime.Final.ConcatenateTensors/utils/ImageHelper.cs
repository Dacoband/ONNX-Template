using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxRuntime.Final.ConcatenateTensors.utils
{
    public static class ImageHelper
    {
        public static Tensor<float> GetImageTensorFromPath(string imageFilePath, int imgWidth = 256, int imgHeight = 256)
        {
            //int batchSize = imageFilePath.Length;
            //Tensor<float> input = new DenseTensor<float>(new[] { batchSize, 9, imgHeight, imgWidth});
            //var mean = new[] { 0.485f, 0.456f, 0.406f };
            //var stddev = new[] { 0.229f, 0.224f, 0.225f };
            //for (int i = 0; i < batchSize; i++)
            //{
            //    using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);
            //    image.Mutate(x =>
            //    {
            //        x.Resize(new ResizeOptions
            //        {
            //            Size = new Size(imgWidth, imgHeight),
            //            Mode = ResizeMode.Crop
            //        });
            //    });
            //    for (int y = 0; y < imgHeight; y++)
            //    {
            //        Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
            //        for (int x = 0; x < image.Width; x++)
            //        {
            //            float normalizedR = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
            //            float normalizedG = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
            //            float normalizedB = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];

            //            for (int c = 0; c < 9; c += 3)
            //            {
            //                input[i, c, y, x] = normalizedR;
            //                input[i, c + 1, y, x] = normalizedG;
            //                input[i, c + 2, y, x] = normalizedB;
            //            }
            //        }

            //    }
            //}

            //// Read image
            //using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            //// Resize image
            //image.Mutate(x =>
            //{
            //    x.Resize(new ResizeOptions
            //    {
            //        Size = new Size(imgWidth, imgHeight),
            //        Mode = ResizeMode.Crop
            //    });
            //});


            //Tensor<float> input = new DenseTensor<float>(new[] { batchSize, 9, imgHeight, imgWidth });

            //var mean = new[] { 0.485f, 0.456f, 0.406f };
            //var stddev = new[] { 0.229f, 0.224f, 0.225f };

            //for (int y = 0; y < image.Height; y++)
            //{
            //    Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
            //    for (int x = 0; x < image.Width; x++)
            //    {

            //        float normalizedR = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
            //        float normalizedG = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
            //        float normalizedB = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];


            //        for (int b = 0; b < batchSize; b++)
            //        {
            //            input[b, 0, y, x] = normalizedR; 
            //            input[b, 1, y, x] = normalizedG; 
            //            input[b, 2, y, x] = normalizedB; 
            //            input[b, 3, y, x] = normalizedR; 
            //            input[b, 4, y, x] = normalizedG; 
            //            input[b, 5, y, x] = normalizedB; 
            //            input[b, 6, y, x] = normalizedR; 
            //            input[b, 7, y, x] = normalizedG; 
            //            input[b, 8, y, x] = normalizedB; 
            //        }
            //    }
            //}

            // Read image
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(imgWidth, imgHeight),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 256, 256 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;
        }
    }
}

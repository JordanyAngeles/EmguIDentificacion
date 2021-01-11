using System;
using System.Diagnostics;
using Emgu.CV.Structure;

namespace Emgu.CV
{
   /// <summary>
   /// Un reconocedor de objetos utilizando PCA (Principle Components Analysis).
   /// </summary>
   [Serializable]
   public class EigenObjectRecognizer
   {
      private Image<Gray, Single>[] _eigenImages;
      private Image<Gray, Single> _avgImage;
      private Matrix<float>[] _eigenValues;
      private string[] _labels;
      private double _eigenDistanceThreshold;

      /// <summary>
      /// Obtener los vectores propios que forman el espacio eigen.
      /// </summary>
      /// <remarks>El método de configuración se usa principalmente para la deserialización, no intente configurarlo a menos que sepa lo que está haciendo.</remarks>
      public Image<Gray, Single>[] EigenImages
      {
         get { return _eigenImages; }
         set { _eigenImages = value; }
      }

      /// <summary>
      /// Obtén o establece las etiquetas para la imagen de entrenamiento correspondiente.
      /// </summary>
      public String[] Labels
      {
         get { return _labels; }
         set { _labels = value; }
      }

      /// <summary>
      /// Obtener o establecer el umbral de distancia eigen.
      /// Cuanto menor sea el número, más probable será que una imagen examinada se trate como un objeto no reconocido. 
      /// /// Ajústelo a un número enorme (por ejemplo, 5000) y el reconocedor siempre tratará la imagen examinada como uno de los objetos conocidos. 
      /// </summary>
      public double EigenDistanceThreshold
      {
         get { return _eigenDistanceThreshold; }
         set { _eigenDistanceThreshold = value; }
      }

      /// <summary>
      /// Obtener la imagen media. 
      /// </summary>
      /// <remarks>El método set se usa principalmente para la deserialización, no intente configurarlo a menos que sepa lo que está haciendo.</remarks>
      public Image<Gray, Single> AverageImage
      {
         get { return _avgImage; }
         set { _avgImage = value; }
      }

      /// <summary>
      /// Obtén los valores propios de cada una de las imágenes de entrenamiento.
      /// </summary>
      /// <remarks>El método set se usa principalmente para la deserialización, no intente configurarlo a menos que sepa lo que está haciendo.</remarks>
      public Matrix<float>[] EigenValues
      {
         get { return _eigenValues; }
         set { _eigenValues = value; }
      }

      private EigenObjectRecognizer()
      {
      }


      /// <summary>
      /// Cree un reconocedor de objetos utilizando los datos y parámetros de transmisión específicos, siempre devolverá el objeto más similar.
      /// </summary>
      /// <param name="images">Las imágenes utilizadas para el entrenamiento, cada una de ellas debe ser del mismo tamaño. Se recomienda que las imágenes estén normalizadas de histograma.</param>
      /// <param name="termCrit">Los criterios para la formación del reconocedor.</param>
      public EigenObjectRecognizer(Image<Gray, Byte>[] images, ref MCvTermCriteria termCrit)
         : this(images, GenerateLabels(images.Length), ref termCrit)
      {
      }

      private static String[] GenerateLabels(int size)
      {
         String[] labels = new string[size];
         for (int i = 0; i < size; i++)
            labels[i] = i.ToString();
         return labels;
      }

      /// <summary>
      /// Crear un reconocedor de objetos utilizando los datos y parámetros de orientación específicos, siempre devolverá el objeto más similar.
      /// </summary>
      /// <param name="images">Las imágenes utilizadas para el entrenamiento, cada una de ellas debe ser del mismo tamaño. Se recomienda que las imágenes estén normalizadas de histograma.</param>
      /// <param name="labels">Las etiquetas correspondientes a las imágenes.</param>
      /// <param name="termCrit">Los criterios para la formación del reconocedor.</param>
      public EigenObjectRecognizer(Image<Gray, Byte>[] images, String[] labels, ref MCvTermCriteria termCrit)
         : this(images, labels, 0, ref termCrit)
      {
      }

        /// <summary>
        ///  Cree un reconocedor de objetos utilizando los datos y parámetros específicos de la instrucción.
        /// </summary>
        /// <param name="images">Las imágenes utilizadas para el entrenamiento, cada una de ellas deben ser del mismo tamaño.</param>
        /// <param name="labels">Las etiquetas correspondientes a las imágenes.</param>
        /// <param name="eigenDistanceThreshold">
        /// El umbral de distancia propio, (0, ~ 1000].
        /// Cuanto menor sea el número, más probable será que una imagen examinada se trate como un objeto no reconocido. 
        /// Si el umbral es & lt; 0, el reconocedor siempre tratará la imagen examinada como uno de los objetos conocidos.
        /// </param>
        /// <param name="termCrit">Los criterios para la formación del reconocedor.</param>
        public EigenObjectRecognizer(Image<Gray, Byte>[] images, String[] labels, double eigenDistanceThreshold, ref MCvTermCriteria termCrit)
      {
         Debug.Assert(images.Length == labels.Length, "The number of images should equals the number of labels");
         Debug.Assert(eigenDistanceThreshold >= 0.0, "Eigen-distance threshold should always >= 0.0");

         CalcEigenObjects(images, ref termCrit, out _eigenImages, out _avgImage);

         /*
         _avgImage.SerializationCompressionRatio = 9;

         foreach (Image<Gray, Single> img in _eigenImages)
             //Establecer la relación de compresión a la mejor compresión. El objeto serializado por lo tanto puede guardar espacios.
             img.SerializationCompressionRatio = 9;
         */

         _eigenValues = Array.ConvertAll<Image<Gray, Byte>, Matrix<float>>(images,
             delegate(Image<Gray, Byte> img)
             {
                return new Matrix<float>(EigenDecomposite(img, _eigenImages, _avgImage));
             });

         _labels = labels;

         _eigenDistanceThreshold = eigenDistanceThreshold;
      }

      #region static methods
      /// <summary>
      /// Cacule las imágenes propias para la imagen de formación específica.
      /// </summary>
      /// <param name="trainingImages">Las imágenes utilizadas para el entrenamiento. </param>
      /// <param name="termCrit">Los criterios para la formación.</param>
      /// <param name="eigenImages">Las imágenes propias resultantes.</param>
      /// <param name="avg">La imagen media resultante</param>
      public static void CalcEigenObjects(Image<Gray, Byte>[] trainingImages, ref MCvTermCriteria termCrit, out Image<Gray, Single>[] eigenImages, out Image<Gray, Single> avg)
      {
         int width = trainingImages[0].Width;
         int height = trainingImages[0].Height;

         IntPtr[] inObjs = Array.ConvertAll<Image<Gray, Byte>, IntPtr>(trainingImages, delegate(Image<Gray, Byte> img) { return img.Ptr; });

         if (termCrit.max_iter <= 0 || termCrit.max_iter > trainingImages.Length)
            termCrit.max_iter = trainingImages.Length;
         
         int maxEigenObjs = termCrit.max_iter;

         #region initialize eigen images
         eigenImages = new Image<Gray, float>[maxEigenObjs];
         for (int i = 0; i < eigenImages.Length; i++)
            eigenImages[i] = new Image<Gray, float>(width, height);
         IntPtr[] eigObjs = Array.ConvertAll<Image<Gray, Single>, IntPtr>(eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; });
         #endregion

         avg = new Image<Gray, Single>(width, height);

         CvInvoke.cvCalcEigenObjects(
             inObjs,
             ref termCrit,
             eigObjs,
             null,
             avg.Ptr);
      }

      /// <summary>
      /// Descomponer la imagen como valores propios, utilizando los vectores propios específicos
      /// </summary>
      /// <param name="src">Descomponer la imagen como valores propios, utilizando los vectores propios específicos</param>
      /// <param name="eigenImages">Las imágenes propias.</param>
      /// <param name="avg">Las imágenes medias.</param>
      /// <returns>Valores propios de la imagen descompuesta.</returns>
      public static float[] EigenDecomposite(Image<Gray, Byte> src, Image<Gray, Single>[] eigenImages, Image<Gray, Single> avg)
      {
         return CvInvoke.cvEigenDecomposite(
             src.Ptr,
             Array.ConvertAll<Image<Gray, Single>, IntPtr>(eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; }),
             avg.Ptr);
      }
      #endregion

      /// <summary>
      /// Dado el valor propio, reconstruir la imagen proyectada.
      /// </summary>
      /// <param name="eigenValue">Los valores propios.</param>
      /// <returns>La imagen proyectada</returns>
      public Image<Gray, Byte> EigenProjection(float[] eigenValue)
      {
         Image<Gray, Byte> res = new Image<Gray, byte>(_avgImage.Width, _avgImage.Height);
         CvInvoke.cvEigenProjection(
             Array.ConvertAll<Image<Gray, Single>, IntPtr>(_eigenImages, delegate(Image<Gray, Single> img) { return img.Ptr; }),
             eigenValue,
             _avgImage.Ptr,
             res.Ptr);
         return res;
      }

      /// <summary>
      /// Obtener la propia-distancia euclidiana entre. <paramref name="image"/> y cada otra imagen en la base de datos
      /// </summary>
      /// <param name="image">La imagen a comparar a partir de las imágenes de entrenamiento.</param>
      /// <returns>Una matriz de distancia propia de cada imagen en las imágenes de entrenamiento</returns>
      public float[] GetEigenDistances(Image<Gray, Byte> image)
      {
         using (Matrix<float> eigenValue = new Matrix<float>(EigenDecomposite(image, _eigenImages, _avgImage)))
            return Array.ConvertAll<Matrix<float>, float>(_eigenValues,
                delegate(Matrix<float> eigenValueI)
                {
                   return (float)CvInvoke.cvNorm(eigenValue.Ptr, eigenValueI.Ptr, Emgu.CV.CvEnum.NORM_TYPE.CV_L2, IntPtr.Zero);
                });
      }

        /// <summary>
        /// Dado que <paramref name="image"/> para ser examinado, encuentre en la base de datos el objeto más similar, devuelva el índice y la distancia propia.
        /// </summary>
        /// <param name="image">La imagen a buscar desde la base de datos</param>
        /// <param name="index">El índice del objeto más similar</param>
        /// <param name="eigenDistance">La distancia propia del objeto más similar</param>
        /// <param name="label">La etiqueta de la imagen específica</param>
        public void FindMostSimilarObject(Image<Gray, Byte> image, out int index, out float eigenDistance, out String label)
      {
         float[] dist = GetEigenDistances(image);

         index = 0;
         eigenDistance = dist[0];
         for (int i = 1; i < dist.Length; i++)
         {
            if (dist[i] < eigenDistance)
            {
               index = i;
               eigenDistance = dist[i];
            }
         }
         label = Labels[index];
      }

      /// <summary>
      /// Intenta reconocer la imagen y devolver su etiqueta.
      /// </summary>
      /// <param name="image">La imagen a reconocer</param>
      /// <returns>
      /// String.Empty, si no se reconoce;
      /// Etiqueta de la imagen correspondiente, de lo contrario.
      /// </returns>
      public String Recognize(Image<Gray, Byte> image)
      {
         int index;
         float eigenDistance;
         String label;
         FindMostSimilarObject(image, out index, out eigenDistance, out label);

         return (_eigenDistanceThreshold <= 0 || eigenDistance < _eigenDistanceThreshold )  ? _labels[index] : String.Empty;
      }
   }
}

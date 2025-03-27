#pragma once

#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>

#include <iostream>
#include <vector>

namespace anima
{

    template <typename TInputPixelType, typename TOutputPixelType>
    class ODFEstimatorCSDImageFilter : public itk::ImageToImageFilter<itk::Image<TInputPixelType, 3>, itk::VectorImage<TOutputPixelType, 3>>
    {
    public:
        /** Standard class typedefs. */
        typedef ODFEstimatorCSDImageFilter Self;
        typedef itk::Image<TInputPixelType, 3> Input3DImageType;
        typedef itk::Image<TInputPixelType, 4> Input4DImageType;
        typedef itk::Image<TOutputPixelType, 3> OutputScalarImageType;
        typedef itk::VectorImage<TOutputPixelType, 3> OutputVectorImageType;
        typedef itk::ImageToImageFilter<Input3DImageType, OutputVectorImageType> Superclass;
        typedef itk::SmartPointer<Self> Pointer;
        typedef itk::SmartPointer<const Self> ConstPointer;

        /** Method for creation through the object factory. */
        itkNewMacro(Self);

        /** Run-time type information (and related methods) */
        itkTypeMacro(ODFEstimatorCSDImageFilter, ImageToImageFilter);

        typedef typename Input3DImageType::Pointer InputImagePointer;
        typedef typename OutputVectorImageType::Pointer OutputVectorImagePointer;
        typedef typename OutputScalarImageType::Pointer OutputScalarImagePointer;
        typedef typename OutputVectorImageType::PixelType OutputVectorImagePixelType;

        /** Superclass typedefs. */
        typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

        void AddGradientDirection(unsigned int i, vnl_vector_fixed<double,3> &grad);
        void SetBValuesList(std::vector<double> bValuesList) { m_BValuesList = bValuesList; }
        OutputVectorImagePointer GetDtiImage() { return m_DtiImage; }
        void SetDtiImage(OutputVectorImagePointer dtiImage) { m_DtiImage = dtiImage; }
        void SetFaImage(OutputScalarImagePointer faImage) { m_FaImage = faImage; }

        itkSetMacro(BValueShellSelected, int);
        itkSetMacro(Lambda, double);
        itkSetMacro(Tau, double);
        itkSetMacro(LOrder, unsigned int);
        itkSetMacro(nbBestVoxel, unsigned int);


    protected:
        ODFEstimatorCSDImageFilter()
        {
            m_GradientDirections.clear();
            //m_PVector.clear();
            m_ReferenceB0Image = nullptr;

            m_BValueShellSelected = -1;
            m_BValueShellTolerance = 20;

            m_Lambda = 1;
            m_LOrder = 8;

            m_SphereSHSampling.clear();
        }

        virtual ~ODFEstimatorCSDImageFilter() {}

        void GenerateOutputInformation() override;
        void BeforeThreadedGenerateData() override;
        void DynamicThreadedGenerateData(const OutputImageRegionType &outputRegionForThread) override;
        void GenerateInitialResponseFunction(unsigned int vectorLength);

    private:
        ITK_DISALLOW_COPY_AND_ASSIGN(ODFEstimatorCSDImageFilter);

        std::vector<vnl_vector_fixed<double,3>> m_GradientDirections;
        std::vector<double> m_BValuesList;
        InputImagePointer m_ReferenceB0Image;

        OutputScalarImagePointer m_EstimatedVarianceImage;
        OutputScalarImagePointer m_EstimatedB0Image;

        OutputVectorImagePointer m_DtiImage;
        OutputScalarImagePointer m_FaImage;

        int m_BValueShellSelected;
        double m_BValueShellTolerance;
        std::vector<unsigned int> m_SelectedDWIIndexes;

        vnl_matrix<double> m_TMatrix; // evaluation matrix computed once and for all before threaded generate data
        vnl_matrix<double> m_BMatrix;
        vnl_matrix<double> m_ResponseFunction;

        std::vector<unsigned int> m_B0Indexes, m_GradientIndexes;

        bool m_Normalize;
        std::string m_FileNameSphereTesselation;
        std::vector<std::vector<double>> m_SphereSHSampling;

        double m_Lambda;
        double m_Tau;
        unsigned int m_LOrder;
        unsigned int m_nbBestVoxel;
    };

} // end of namespace anima

#include "animaODFEstimatorCSDImageFilter.hxx"

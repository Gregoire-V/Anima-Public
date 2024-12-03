#pragma once

#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>

#include <iostream>
#include <vector>

namespace anima
{

    template <typename TInputPixelType, typename TOutputPixelType>
    class ODFEstimatorImageFilter : public itk::ImageToImageFilter<itk::Image<TInputPixelType, 3>, itk::VectorImage<TOutputPixelType, 3>>
    {
    public:
        /** Standard class typedefs. */
        typedef ODFEstimatorImageFilter Self;
        typedef itk::Image<TInputPixelType, 3> TInputImage;
        typedef itk::Image<TInputPixelType, 4> Image4DType;
        //typedef itk::Image<TOutputPixelType, 3> OutputScalarImageType;
        typedef itk::VectorImage<TOutputPixelType, 3> TOutputImage;
        typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
        typedef itk::SmartPointer<Self> Pointer;
        typedef itk::SmartPointer<const Self> ConstPointer;

        /** Method for creation through the object factory. */
        itkNewMacro(Self);

        /** Run-time type information (and related methods) */
        itkTypeMacro(ODFEstimatorImageFilter, ImageToImageFilter);

        typedef typename TInputImage::Pointer InputImagePointer;
        typedef typename TOutputImage::Pointer OutputImagePointer;
        //typedef typename OutputScalarImageType::Pointer OutputScalarImagePointer;

        /** Superclass typedefs. */
        typedef typename Superclass::OutputImageRegionType OutputImageRegionType;

        void AddGradientDirection(unsigned int i, std::vector<double> &grad);
        void SetBValuesList(std::vector<double> bValuesList) { m_BValuesList = bValuesList; }
        itkSetMacro(BValueShellSelected, int);

        itkSetMacro(Lambda, double);
        itkSetMacro(Tau, double);
        itkSetMacro(LOrder, unsigned int);


    protected:
        ODFEstimatorImageFilter()
        {
            m_GradientDirections.clear();
            m_PVector.clear();
            m_ReferenceB0Image = nullptr;

            m_BValueShellSelected = -1;
            m_BValueShellTolerance = 20;

            m_Lambda = 1;
            m_LOrder = 8;

            m_SphereSHSampling.clear();
        }

        virtual ~ODFEstimatorImageFilter() {}

        void GenerateOutputInformation() override;
        void BeforeThreadedGenerateData() override;
        void DynamicThreadedGenerateData(const OutputImageRegionType &outputRegionForThread) override;

    private:
        ITK_DISALLOW_COPY_AND_ASSIGN(ODFEstimatorImageFilter);

        std::vector<std::vector<double>> m_GradientDirections;
        std::vector<double> m_BValuesList;
        InputImagePointer m_ReferenceB0Image;

        OutputScalarImagePointer m_EstimatedVarianceImage;
        OutputScalarImagePointer m_EstimatedB0Image;

        int m_BValueShellSelected;
        double m_BValueShellTolerance;
        std::vector<unsigned int> m_SelectedDWIIndexes;

        vnl_matrix<double> m_TMatrix; // evaluation matrix computed once and for all before threaded generate data
        vnl_matrix<double> m_BMatrix;
        std::vector<double> m_DeconvolutionVector;
        std::vector<double> m_PVector;

        std::vector<unsigned int> m_B0Indexes, m_GradientIndexes;

        bool m_Normalize;
        std::string m_FileNameSphereTesselation;
        std::vector<std::vector<double>> m_SphereSHSampling;

        double m_Lambda;
        double m_SharpnessRatio; // See Descoteaux et al. TMI 2009, article plus appendix
        bool m_Sharpen;
        bool m_UseAganjEstimation;
        double m_DeltaAganjRegularization;
        unsigned int m_LOrder;
    };

} // end of namespace anima

#include "animaODFEstimatorCSDImageFilter.hxx"

#pragma once

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMultiResolutionPyramidImageFilter.h>

#include <itkVectorResampleImageFilter.h>

#include <animaVelocityUtils.h>
#include <animaLinearTransformEstimationTools.h>

#include <animaAsymmetricBlockMatchingRegistrationMethod.h>
#include <animaSymmetricBlockMatchingRegistrationMethod.h>
#include <animaKissingSymmetricBlockMatchingRegistrationMethod.h>

// ------------------------------

namespace anima
{

template <unsigned int ImageDimension>
PyramidalDenseSVFMatchingBridge<ImageDimension>::PyramidalDenseSVFMatchingBridge()
{
    m_ReferenceImage = NULL;
    m_FloatingImage = NULL;

    m_OutputTransform = BaseTransformType::New();
    m_OutputTransform->SetIdentity();

    m_outputTransformFile = "";

    m_OutputImage = NULL;

    m_BlockSize = 5;
    m_BlockSpacing = 2;
    m_StDevThreshold = 5;

    m_SymmetryType = Asymmetric;
    m_Transform = Translation;
    m_Metric = SquaredCorrelation;
    m_Optimizer = Bobyqa;

    m_MaximumIterations = 10;
    m_MinimalTransformError = 0.01;
    m_OptimizerMaximumIterations = 100;
    m_SearchRadius = 2;
    m_SearchAngleRadius = 5;
    m_SearchSkewRadius = 5;
    m_SearchScaleRadius = 0.1;
    m_FinalRadius = 0.001;
    m_StepSize = 1;
    m_TranslateUpperBound = 50;
    m_AngleUpperBound = 180;
    m_SkewUpperBound = 45;
    m_ScaleUpperBound = 3;
    m_Agregator = Baloo;
    m_ExtrapolationSigma = 3;
    m_ElasticSigma = 3;
    m_OutlierSigma = 3;
    m_MEstimateConvergenceThreshold = 0.01;
    m_NeighborhoodApproximation = 2.5;
    m_UseTransformationDam = true;
    m_DamDistance = 2.5;
    m_NumberOfPyramidLevels = 3;
    m_LastPyramidLevel = 0;
    m_PercentageKept = 0.8;
    this->SetNumberOfThreads(itk::MultiThreader::GetGlobalDefaultNumberOfThreads());

    m_Abort = false;

    m_callback = itk::CStyleCommand::New();
    m_callback->SetClientData ((void *) this);
    m_callback->SetCallback (ManageProgress);
}

template <unsigned int ImageDimension>
PyramidalDenseSVFMatchingBridge<ImageDimension>::~PyramidalDenseSVFMatchingBridge()
{
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::Abort()
{
    m_Abort = true;

    if(m_bmreg)
        m_bmreg->Abort();
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::InitializeBlocksOnImage(InitializerPointer &initPtr, InputImageType *image)
{
    // Init blocks
    initPtr = InitializerType::New();
    initPtr->AddReferenceImage(image);

    if (this->GetNumberOfThreads() != 0)
        initPtr->SetNumberOfThreads(this->GetNumberOfThreads());

    initPtr->SetPercentageKept(m_PercentageKept);
    initPtr->SetBlockSize(m_BlockSize);
    initPtr->SetBlockSpacing(m_BlockSpacing);
    initPtr->SetScalarVarianceThreshold(m_StDevThreshold * m_StDevThreshold);

    initPtr->SetRequestedRegion(image->GetLargestPossibleRegion());
    initPtr->SetComputeOuterDam(m_UseTransformationDam);
    initPtr->SetDamDistance(m_DamDistance * m_ExtrapolationSigma);
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::Update()
{
    m_Abort = false;

    // progress management
    m_progressReporter = new itk::ProgressReporter(this, 0, GetNumberOfPyramidLevels()*m_MaximumIterations);
    this->AddObserver(itk::ProgressEvent(), m_progressCallback);

    this->InvokeEvent(itk::StartEvent());

    this->SetupPyramids();

    // Iterate over pyramid levels
    for (unsigned int i = 0;i < m_ReferencePyramid->GetNumberOfLevels();++i)
    {
        if (i + m_LastPyramidLevel >= m_ReferencePyramid->GetNumberOfLevels())
            continue;

        typename InputImageType::Pointer refImage = m_ReferencePyramid->GetOutput(i);
        refImage->DisconnectPipeline();

        typename InputImageType::Pointer floImage = m_FloatingPyramid->GetOutput(i);
        floImage->DisconnectPipeline();

        // Update field to match the current resolution
        if (m_OutputTransform->GetParametersAsVectorField() != NULL)
        {
            typedef itk::VectorResampleImageFilter<VelocityFieldType,VelocityFieldType> VectorResampleFilterType;
            typedef typename VectorResampleFilterType::Pointer VectorResampleFilterPointer;

            AffineTransformPointer tmpIdentity = AffineTransformType::New();
            tmpIdentity->SetIdentity();

            VectorResampleFilterPointer tmpResample = VectorResampleFilterType::New();
            tmpResample->SetTransform(tmpIdentity);
            tmpResample->SetInput(m_OutputTransform->GetParametersAsVectorField());

            tmpResample->SetSize(refImage->GetLargestPossibleRegion().GetSize());
            tmpResample->SetOutputOrigin(refImage->GetOrigin());
            tmpResample->SetOutputSpacing(refImage->GetSpacing());
            tmpResample->SetOutputDirection(refImage->GetDirection());

            tmpResample->Update();

            VelocityFieldType *tmpOut = tmpResample->GetOutput();
            m_OutputTransform->SetParametersAsVectorField(tmpOut);
            tmpOut->DisconnectPipeline();
        }

        std::cout << "Processing pyramid level " << i << std::endl;
        std::cout << "Image size: " << refImage->GetLargestPossibleRegion().GetSize() << std::endl;

        double meanSpacing = 0;
        for (unsigned int j = 0;j < ImageDimension;++j)
            meanSpacing += refImage->GetSpacing()[j];
        meanSpacing /= ImageDimension;

        // Init agregator mean shift parameters
        BaseAgregatorType* agregPtr = NULL;

        if (m_Agregator == MSmoother)
        {
            MEstimateAgregatorType *agreg = new MEstimateAgregatorType;
            agreg->SetExtrapolationSigma(m_ExtrapolationSigma * meanSpacing);
            agreg->SetOutlierRejectionSigma(m_OutlierSigma);
            agreg->SetOutputTransformType(BaseAgregatorType::SVF);

            if (this->GetNumberOfThreads() != 0)
                agreg->SetNumberOfThreads(this->GetNumberOfThreads());

            agreg->SetGeometryInformation(refImage.GetPointer());

            agreg->SetNeighborhoodHalfSize((unsigned int)floor(m_ExtrapolationSigma * m_NeighborhoodApproximation));
            agreg->SetDistanceBoundary(m_ExtrapolationSigma * meanSpacing * m_NeighborhoodApproximation);
            agreg->SetMEstimateConvergenceThreshold(m_MEstimateConvergenceThreshold);

            agregPtr = agreg;
        }
        else
        {
            BalooAgregatorType *agreg = new BalooAgregatorType;
            agreg->SetExtrapolationSigma(m_ExtrapolationSigma * meanSpacing);
            agreg->SetOutlierRejectionSigma(m_OutlierSigma);
            agreg->SetOutputTransformType(BaseAgregatorType::SVF);

            if (this->GetNumberOfThreads() != 0)
                agreg->SetNumberOfThreads(this->GetNumberOfThreads());

            agreg->SetGeometryInformation(refImage.GetPointer());

            agregPtr = agreg;
        }

        // Init matcher
        switch (m_SymmetryType)
        {
            case Asymmetric:
            {
                typedef typename anima::AsymmetricBlockMatchingRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                m_bmreg = BlockMatchRegistrationType::New();

                typename InitializerType::Pointer initPtr;
                this->InitializeBlocksOnImage(initPtr, refImage);

                m_bmreg->SetBlockRegions(initPtr->GetOutput());
                if (m_Agregator == MSmoother)
                {
                    MEstimateAgregatorType *agreg = dynamic_cast <MEstimateAgregatorType *> (agregPtr);
                    agreg->SetDamIndexes(initPtr->GetDamIndexes());
                }
                else
                {
                    BalooAgregatorType *agreg = dynamic_cast <BalooAgregatorType *> (agregPtr);
                    agreg->SetDamIndexes(initPtr->GetDamIndexes());
                }

                std::cout << "Generated " << initPtr->GetOutput().size() << " blocks..." << std::endl;

                break;
            }

            case Symmetric:
            {
                typedef typename anima::SymmetricBlockMatchingRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                typename BlockMatchRegistrationType::Pointer tmpReg = BlockMatchRegistrationType::New();

                typename InitializerType::Pointer initPtr;
                this->InitializeBlocksOnImage(initPtr, refImage);

                tmpReg->SetFixedBlockRegions(initPtr->GetOutput());
                tmpReg->SetFixedDamIndexes(initPtr->GetDamIndexes());
                std::cout << "Generated " << initPtr->GetOutput().size() << " blocks..." << std::endl;

                this->InitializeBlocksOnImage(initPtr, floImage);

                tmpReg->SetMovingBlockRegions(initPtr->GetOutput());
                tmpReg->SetMovingDamIndexes(initPtr->GetDamIndexes());
                std::cout << "Generated " << initPtr->GetOutput().size() << " blocks..." << std::endl;

                m_bmreg = tmpReg;
                break;
            }

            case Kissing:
            {
                typedef typename anima::KissingSymmetricBlockMatchingRegistrationMethod <InputImageType> BlockMatchRegistrationType;
                typename BlockMatchRegistrationType::Pointer tmpReg = BlockMatchRegistrationType::New();

                tmpReg->SetBlockPercentageKept(GetPercentageKept());
                tmpReg->SetBlockSize(GetBlockSize());
                tmpReg->SetBlockSpacing(GetBlockSpacing());
                tmpReg->SetUseTransformationDam(m_UseTransformationDam);
                tmpReg->SetDamDistance(m_DamDistance * m_ExtrapolationSigma);

                m_bmreg = tmpReg;

                break;
            }
        }

        if (m_progressCallback)
        {
            // we cannot connect directly bmreg to m_progressCallback
            // we need to create a new progressReporter with more iterations (m_progressReporter),
            // to listen to progress events from bmreg and to send new ones to m_progressCallback
            m_bmreg->AddObserver(itk::ProgressEvent(), m_callback);
        }

        m_bmreg->SetBlockScalarVarianceThreshold(GetStDevThreshold() * GetStDevThreshold());
        m_bmreg->SetAgregator(agregPtr);

        if (this->GetNumberOfThreads() != 0)
            m_bmreg->SetNumberOfThreads(this->GetNumberOfThreads());

        m_bmreg->SetFixedImage(refImage);
        m_bmreg->SetMovingImage(floImage);

        m_bmreg->SetSVFElasticRegSigma(m_ElasticSigma * meanSpacing);

        switch (m_Transform)
        {
            case Translation:
                m_bmreg->SetTransformKind(BaseBlockMatchRegistrationType::Translation);
                agregPtr->SetInputTransformType(BaseAgregatorType::TRANSLATION);
                break;
            case Rigid:
                m_bmreg->SetTransformKind(BaseBlockMatchRegistrationType::Rigid);
                agregPtr->SetInputTransformType(BaseAgregatorType::RIGID);
                break;
            case Affine:
            default:
                m_bmreg->SetTransformKind(BaseBlockMatchRegistrationType::Affine);
                agregPtr->SetInputTransformType(BaseAgregatorType::AFFINE);
                break;
        }

        switch (m_Optimizer)
        {
            case Exhaustive:
                m_bmreg->SetOptimizerKind(BaseBlockMatchRegistrationType::Exhaustive);
                break;
            case Bobyqa:
            default:
                m_bmreg->SetOptimizerKind(BaseBlockMatchRegistrationType::Bobyqa);
                break;
        }

        switch (m_Metric)
        {
            case SquaredCorrelation:
                m_bmreg->SetMetricKind(BaseBlockMatchRegistrationType::SquaredCorrelation);
                break;
            case Correlation:
                m_bmreg->SetMetricKind(BaseBlockMatchRegistrationType::Correlation);
                break;
            case MeanSquares:
            default:
                m_bmreg->SetMetricKind(BaseBlockMatchRegistrationType::MeanSquares);
                break;
        }

        m_bmreg->SetMaximumIterations(m_MaximumIterations);
        m_bmreg->SetMinimalTransformError(m_MinimalTransformError);
        m_bmreg->SetOptimizerMaximumIterations(m_OptimizerMaximumIterations);

        m_bmreg->SetInitialTransform(m_OutputTransform.GetPointer());

        double sr = m_SearchRadius;
        m_bmreg->SetSearchRadius(sr);

        double sar = m_SearchAngleRadius;
        m_bmreg->SetSearchAngleRadius(sar);

        double skr = m_SearchSkewRadius;
        m_bmreg->SetSearchSkewRadius(skr);

        double scr = m_SearchScaleRadius;
        m_bmreg->SetSearchScaleRadius(scr);

        double fr = m_FinalRadius;
        m_bmreg->SetFinalRadius(fr);

        double ss = m_StepSize;
        m_bmreg->SetStepSize(ss);

        double tub = m_TranslateUpperBound;
        m_bmreg->SetTranslateMax(tub);

        double aub = m_AngleUpperBound;
        m_bmreg->SetAngleMax(aub);

        double skub = m_SkewUpperBound;
        m_bmreg->SetSkewMax(skub);

        double scub = m_ScaleUpperBound;
        m_bmreg->SetScaleMax(scub);

        try
        {
            m_bmreg->Update();
            std::cout << "Block Matching Registration stop condition "
                      << m_bmreg->GetStopConditionDescription()
                      << std::endl;
        }
        catch( itk::ExceptionObject & err )
        {
            std::cout << "ExceptionObject caught !" << err << std::endl;
            exit(-1);
        }

        const BaseTransformType *resTrsf = dynamic_cast <const BaseTransformType *> (m_bmreg->GetOutput()->Get());
        m_OutputTransform->SetParametersAsVectorField(resTrsf->GetParametersAsVectorField());
    }

    if (m_Abort)
        std::cout << "Process aborted" << std::endl;

    this->InvokeEvent(itk::EndEvent());

    if (m_SymmetryType == Kissing)
    {
        VelocityFieldType *finalTrsfField = const_cast <VelocityFieldType *> (m_OutputTransform->GetParametersAsVectorField());
        typedef itk::MultiplyImageFilter <VelocityFieldType,itk::Image <float,ImageDimension>, VelocityFieldType> MultiplyFilterType;

        typename MultiplyFilterType::Pointer fieldMultiplier = MultiplyFilterType::New();
        fieldMultiplier->SetInput(finalTrsfField);
        fieldMultiplier->SetConstant(2.0);
        fieldMultiplier->SetNumberOfThreads(GetNumberOfThreads());
        fieldMultiplier->InPlaceOn();

        fieldMultiplier->Update();

        VelocityFieldType *outputField = fieldMultiplier->GetOutput();
        m_OutputTransform->SetParametersAsVectorField(fieldMultiplier->GetOutput());
        outputField->DisconnectPipeline();
    }

    DisplacementFieldTransformPointer outputDispTrsf = DisplacementFieldTransformType::New();
    anima::GetSVFExponential(m_OutputTransform.GetPointer(), outputDispTrsf.GetPointer(), false);

    typedef typename anima::ResampleImageFilter<InputImageType, InputImageType,
                                                typename BaseAgregatorType::ScalarType> ResampleFilterType;
    typename ResampleFilterType::Pointer tmpResample = ResampleFilterType::New();
    tmpResample->SetTransform(outputDispTrsf);
    tmpResample->SetInput(m_FloatingImage);

    tmpResample->SetSize(m_ReferenceImage->GetLargestPossibleRegion().GetSize());
    tmpResample->SetOutputOrigin(m_ReferenceImage->GetOrigin());
    tmpResample->SetOutputSpacing(m_ReferenceImage->GetSpacing());
    tmpResample->SetOutputDirection(m_ReferenceImage->GetDirection());
    tmpResample->SetDefaultPixelValue(0);
    tmpResample->Update();

    m_OutputImage = tmpResample->GetOutput();
    m_OutputImage->DisconnectPipeline();
}

template <unsigned int ImageDimension>
typename PyramidalDenseSVFMatchingBridge<ImageDimension>::DisplacementFieldTransformPointer
PyramidalDenseSVFMatchingBridge<ImageDimension>::GetOutputDisplacementFieldTransform()
{
    DisplacementFieldTransformPointer outputDispTrsf = DisplacementFieldTransformType::New();

    anima::GetSVFExponential(m_OutputTransform.GetPointer(), outputDispTrsf.GetPointer(), false);

    return outputDispTrsf;
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::EmitProgress(int prog)
{
    if (m_progressReporter)
        m_progressReporter->CompletedPixel();
}

template <unsigned int ImageDimension>
void PyramidalDenseSVFMatchingBridge<ImageDimension>::ManageProgress (itk::Object* caller, const itk::EventObject& event, void* clientData)
{
    PyramidalDenseSVFMatchingBridge * source = reinterpret_cast<PyramidalDenseSVFMatchingBridge *> (clientData);
    itk::ProcessObject *processObject = (itk::ProcessObject *) caller;

    if (source && processObject)
        source->EmitProgress(processObject->GetProgress() * 100);
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::WriteOutputs()
{
    std::cout << "Writing output image to: " << m_resultFile << std::endl;

    typename itk::ImageFileWriter <InputImageType>::Pointer imageWriter = itk::ImageFileWriter <InputImageType>::New();
    imageWriter->SetUseCompression(true);
    imageWriter->SetInput(m_OutputImage);
    imageWriter->SetFileName(m_resultFile);

    imageWriter->Update();

    if (m_outputTransformFile != "")
    {
        std::cout << "Writing output SVF to: " << m_outputTransformFile << std::endl;
        typename itk::ImageFileWriter <VelocityFieldType>::Pointer writer = itk::ImageFileWriter <VelocityFieldType>::New();
        writer->SetInput(m_OutputTransform->GetParametersAsVectorField());
        writer->SetFileName(m_outputTransformFile);
        writer->Update();
    }
}

template <unsigned int ImageDimension>
void
PyramidalDenseSVFMatchingBridge<ImageDimension>::SetupPyramids()
{
    // Create pyramid here, check images actually are of the same size.
    m_ReferencePyramid = PyramidType::New();

    m_ReferencePyramid->SetInput(m_ReferenceImage);
    m_ReferencePyramid->SetNumberOfLevels(m_NumberOfPyramidLevels);

    if (this->GetNumberOfThreads() != 0)
        m_ReferencePyramid->SetNumberOfThreads(this->GetNumberOfThreads());

    m_ReferencePyramid->Update();

    // Create pyramid for floating image
    m_FloatingPyramid = PyramidType::New();

    m_FloatingPyramid->SetInput(m_FloatingImage);
    m_FloatingPyramid->SetNumberOfLevels(m_NumberOfPyramidLevels);

    if (this->GetNumberOfThreads() != 0)
        m_FloatingPyramid->SetNumberOfThreads(this->GetNumberOfThreads());

    m_FloatingPyramid->Update();
}

} // end of namespace
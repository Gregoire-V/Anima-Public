#include <itkThresholdImageFilter.h>
#include <itkThresholdLabelerImageFilter.h>
#include <itkImageRegionConstIterator.h>

#include <animaReadWriteFunctions.h>

#include <limits>
#include <iostream>
#include <fstream>
#include <tclap/CmdLine.h>

using namespace std;

template<int nbDim>
int createOutputMaskImage(string inputImageName, string inputMaskImageName, string outputImageName, double thrValue, double upperThr, double adaptThr, bool outputValuesInversion)
{
    typedef itk::Image<double, nbDim> DoubleImageType;
    typedef itk::Image<unsigned char, nbDim> UCImageType;
    
    typedef itk::ImageRegionConstIterator<DoubleImageType> DoubleImageIterator;
    typedef itk::ImageRegionIterator<UCImageType> UCImageIterator;

    typedef itk::ThresholdImageFilter<DoubleImageType> ThresholdFilterType;
    typedef itk::ThresholdLabelerImageFilter<DoubleImageType, UCImageType> LabelerFilterType;

    typename DoubleImageType::Pointer inputImage = anima::readImage <DoubleImageType>(inputImageName);
    
    typename DoubleImageType::RegionType tmpRegionInputImage = inputImage->GetLargestPossibleRegion();
    DoubleImageIterator doubleIt(inputImage, tmpRegionInputImage);

    unsigned int totalSize = 1;

    for (unsigned int i = 0; i < nbDim; i++)
    {
        totalSize *= tmpRegionInputImage.GetSize()[i];
    }

    //std::vector<double> tmpVec(totalSize);

    unsigned int idx=0;
    /*
    if (inputMaskImageName != "")
    {
        typename UCImageType::Pointer maskImage = anima::readImage<UCImageType>(inputMaskImageName);

        typename UCImageType::RegionType tmpRegionMaskImage = maskImage->GetLargestPossibleRegion();
        UCImageIterator ucIt(maskImage,tmpRegionMaskImage);

        for (unsigned int i = 0; i < nbDim; i++)
        {
            if (tmpRegionInputImage.GetSize()[i]!=tmpRegionMaskImage.GetSize()[i])
            {
                std::cerr << "InputImage size != MaskImage size" << std::endl;
                return EXIT_FAILURE;
            }
        }

        for (unsigned int i = 0;i < totalSize;++i)
        {
            if(ucIt.Get() == 1)
            {
                tmpVec[idx] = doubleIt.Get();
                ++idx;
            }

            ++ucIt;
            ++doubleIt;
        }

        tmpVec.resize(idx);
    }
    else
    {
        idx = totalSize;
        for (unsigned int i = 0;i < totalSize;++i)
        {
            tmpVec[i] = doubleIt.Get();
            ++doubleIt;
        }
    }
        */

    if ((adaptThr < 0) || (adaptThr > 1))
    {
        std::cerr << "Adaptative threshold value has to be included in the [0,1] interval" << std::endl;
        return EXIT_FAILURE;
    }
    
    unsigned int partialElt = (unsigned int) floor(adaptThr*idx);
    if (partialElt == idx)
        partialElt = idx - 1;

    //if (partialElt != 0)
        //std::partial_sort(tmpVec.begin(),tmpVec.begin() + partialElt + 1,tmpVec.end());

    //if (partialElt != 0)
        //thrValue = tmpVec[partialElt];

    typename ThresholdFilterType::Pointer thrFilter = ThresholdFilterType::New();
    thrFilter->SetInput(inputImage);

    if (upperThr < thrValue)
        upperThr = thrValue;

    if (upperThr != USHRT_MAX)
        thrFilter->ThresholdOutside(thrValue,upperThr);
    else
        thrFilter->ThresholdBelow(thrValue);
    
    try
    {
        thrFilter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
        std::cerr << e << std::endl;
        return EXIT_FAILURE;
    }
    
    typename LabelerFilterType::Pointer mainFilter = LabelerFilterType::New();
    mainFilter->SetInput(thrFilter->GetOutput());

    typename LabelerFilterType::RealThresholdVector thrVals;
    thrVals.push_back(0);

    mainFilter->SetRealThresholds(thrVals);

    try
    {
        mainFilter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
        std::cerr << e << std::endl;
        return EXIT_FAILURE;
    }
    
    if (outputValuesInversion)
    {
        UCImageIterator resIt(mainFilter->GetOutput(), tmpRegionInputImage);
        
        for (unsigned int i = 0;i < totalSize;++i)
        {
            resIt.Set(1-resIt.Get());
            ++resIt;
        }
    }

    anima::writeImage <UCImageType> (outputImageName, mainFilter->GetOutput());
    return EXIT_SUCCESS;
}





//using x1 = template<int nbCompenent>



int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS/Empenn Team", ' ',ANIMA_VERSION);
    
    TCLAP::ValueArg<std::string> inputArg("i","inputimage","Input image",true,"","Input image",cmd);
    TCLAP::ValueArg<std::string> outputArg("o","outputimage","Output image",true,"","Output image",cmd);
    TCLAP::ValueArg<std::string> maskArg("m","maskfile","mask file",false,"","mask file",cmd);

    TCLAP::ValueArg<double> thrArg("t","thr","Threshold value",false,1.0,"Threshold value",cmd);
    TCLAP::ValueArg<double> upperThrArg("u","uthr","Upper threshold value",false,std::numeric_limits <double>::max(),"Upper threshold value",cmd);
    TCLAP::ValueArg<double> adaptThrArg("a","adaptivethr","Adaptative threshold value (between 0 and 1)",false,0.0,"adaptative threshold value",cmd);
    
    TCLAP::SwitchArg invArg("I","inv","Computes 1-res",cmd,false);
    
    try
    {
        cmd.parse(argc,argv);
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputArg.getValue().c_str(), itk::IOFileModeEnum::ReadMode);
    imageIO->SetFileName(inputArg.getValue());
    imageIO->ReadImageInformation();

    unsigned int nbComponents = imageIO->GetNumberOfComponents();
    unsigned int nbDim = imageIO->GetNumberOfDimensions();

    int exit_value; 

    switch (nbDim)
    {
        case 2:
            exit_value = createOutputMaskImage<2>(inputArg.getValue(), maskArg.getValue(), outputArg.getValue(), thrArg.getValue(), upperThrArg.getValue(), adaptThrArg.getValue(), invArg.isSet());
            break;
        case 3:
            exit_value = createOutputMaskImage<3>(inputArg.getValue(), maskArg.getValue(), outputArg.getValue(), thrArg.getValue(), upperThrArg.getValue(), adaptThrArg.getValue(), invArg.isSet());
            break;
        case 4:
            exit_value = createOutputMaskImage<4>(inputArg.getValue(), maskArg.getValue(), outputArg.getValue(), thrArg.getValue(), upperThrArg.getValue(), adaptThrArg.getValue(), invArg.isSet());
            break;
        default:
            std::cerr<<"Unsupported number of dimensions for input Image (supported values are : 2, 3 and 4)"<<std::endl;
            break;
    }

    return exit_value;
}

    /*
    if (nbComponents == 1)
    {
        //scalar image
    
    }

    else
    {
        //vectorial image
    }
        */

    //typedef itk::Image<double, nbDim> DoubleImageType;
    //typedef itk::Image<unsigned char, nbDim> UCImageType;
    //
    //typedef itk::ImageRegionConstIterator <DoubleImageType> DoubleImageIterator;
    //typedef itk::ImageRegionIterator <UCImageType> UCImageIterator;
//
    //typedef itk::ThresholdImageFilter <DoubleImageType> ThresholdFilterType;
    //typedef itk::ThresholdLabelerImageFilter <DoubleImageType, UCImageType> LabelerFilterType;
//
    //DoubleImageType::Pointer inputImage = anima::readImage <DoubleImageType> (inputArg.getValue());
    //
    //DoubleImageType::RegionType tmpRegionInputImage = inputImage->GetLargestPossibleRegion();
    //DoubleImageIterator doubleIt(inputImage, tmpRegionInputImage);
//
    //for (unsigned int i = 0; i < nbDim; i++)
    //{
    //    totalSize *= tmpRegionInputImage.GetSize()[i];
    //}
//
    //std::vector<double> tmpVec(totalSize);
//
    //unsigned int idx=0;
    //
    //if (maskArg.getValue() != "")
    //{
    //    UCImageType::Pointer maskImage = anima::readImage <UCImageType> (maskArg.getValue());
//
    //    UCImageType::RegionType tmpRegionMaskImage = maskImage->GetLargestPossibleRegion();
    //    UCImageIterator ucIt(maskImage,tmpRegionMaskImage);
//
    //    for (unsigned int i = 0; i < nbDim; i++)
    //    {
    //        if (tmpRegionInputImage.GetSize()[i]!=tmpRegionMaskImage.GetSize()[i])
    //        {
    //            std::cerr << "InputImage size != MaskImage size" << std::endl;
    //            return EXIT_FAILURE;
    //        }
    //    }
//
    //    for (unsigned int i = 0;i < totalSize;++i)
    //    {
    //        if(ucIt.Get() == 1)
    //        {
    //            tmpVec[idx] = doubleIt.Get();
    //            ++idx;
    //        }
//
    //        ++ucIt;
    //        ++doubleIt;
    //    }
//
    //    tmpVec.resize(idx);
    //}
    //else
    //{
    //    idx = totalSize;
    //    for (unsigned int i = 0;i < totalSize;++i)
    //    {
    //        tmpVec[i] = doubleIt.Get();
    //        ++doubleIt;
    //    }
    //}
//
    //if ((adaptThrArg.getValue() < 0) || (adaptThrArg.getValue() > 1))
    //{
    //    std::cerr << "Adaptative threshold value has to be included in the [0,1] interval" << std::endl;
    //    return EXIT_FAILURE;
    //}
    //
    //unsigned int partialElt = (unsigned int) floor(adaptThrArg.getValue()*idx);
    //if (partialElt == idx)
    //    partialElt = idx - 1;
//
    //if (partialElt != 0)
    //    std::partial_sort(tmpVec.begin(),tmpVec.begin() + partialElt + 1,tmpVec.end());
//
    //double thrV = thrArg.getValue();
    //if (partialElt != 0)
    //    thrV = tmpVec[partialElt];
//
    //ThresholdFilterType::Pointer thrFilter = ThresholdFilterType::New();
    //thrFilter->SetInput(inputImage);
//
    //double upperThr = upperThrArg.getValue();
    //if (upperThr < thrV)
    //    upperThr = thrV;
//
    //if (upperThr != USHRT_MAX)
    //    thrFilter->ThresholdOutside(thrV,upperThr);
    //else
    //    thrFilter->ThresholdBelow(thrV);
    //
    //try
    //{
    //    thrFilter->Update();
    //}
    //catch (itk::ExceptionObject &e)
    //{
    //    std::cerr << e << std::endl;
    //    return EXIT_FAILURE;
    //}
    //
    //LabelerFilterType::Pointer mainFilter = LabelerFilterType::New();
    //mainFilter->SetInput(thrFilter->GetOutput());
//
    //LabelerFilterType::RealThresholdVector thrVals;
    //thrVals.push_back(0);
//
    //mainFilter->SetRealThresholds(thrVals);
//
    //try
    //{
    //    mainFilter->Update();
    //}
    //catch (itk::ExceptionObject &e)
    //{
    //    std::cerr << e << std::endl;
    //    return EXIT_FAILURE;
    //}
    //
    //if (invArg.isSet())
    //{
    //    UCImageIterator resIt(mainFilter->GetOutput(), tmpRegionInputImage);
    //    
    //    for (unsigned int i = 0;i < totalSize;++i)
    //    {
    //        resIt.Set(1-resIt.Get());
    //        ++resIt;
    //    }
    //}
//
    //anima::writeImage <UCImageType> (outputArg.getValue(), mainFilter->GetOutput());
//
    

#include <animaGradientFileReader.h>
#include <animaDTIEstimationImageFilter.h>
#include <animaODFEstimatorCSDImageFilter.h>
#include <animaReadWriteFunctions.h>

#include <itkTimeProbe.h>

#include <tclap/CmdLine.h>

int main(int argc, char **argv)
{
    TCLAP::CmdLine cmd("INRIA / IRISA - VisAGeS/Empenn Team", ' ', ANIMA_VERSION);

    TCLAP::ValueArg<std::string> inArg(
        "i", "input",
        "List of diffusion weighted images or 4D volume",
        true, "", "Input diffusion images", cmd);
    TCLAP::ValueArg<std::string> resArg(
        "o", "outputfile",
        "Result ODF image",
        true, "", "Result ODF image", cmd);
    TCLAP::ValueArg<std::string> gradArg(
        "g", "gradientlist",
        "List of gradients (text file)",
        true, "", "List of gradients (text file)", cmd);
    TCLAP::ValueArg<std::string> bvalArg(
        "b", "bval",
        "Input b-values",
        true, "", "Input b-values (text file)", cmd);
    TCLAP::ValueArg<double> lambdaArg(
        "l", "lambda",
        "Lambda regularization parameter (see Tournier CSD 2006)",
        false, 1, "lambda for regularization (real value)", cmd);
    TCLAP::ValueArg<double> tauArg(
        "t", "tau",
        "Tau parameter (see Tournier CSD 2006)",
        false, 0.006, "tau parameter (real value)", cmd);
    TCLAP::ValueArg<unsigned int> orderArg(
        "k", "order",
        "Order of spherical harmonics basis",
        false, 8, "Order of SH basis", cmd);
    TCLAP::SwitchArg bvalueScaleArg(
        "B", "b-no-scale",
        "Do not scale b-values according to gradient norm",
        cmd);
    TCLAP::ValueArg<int> selectedBvalArg(
        "v", "select-bval", 
        "B-value shell used to estimate ODFs (default: first one in data volume above 10)", 
        false, -1, "b-value shell selection", cmd);
    TCLAP::ValueArg<unsigned int> nbpArg(
        "T", "nb-threads",
        "An integer value specifying the number of threads to run on (default: all cores).",
        false, itk::MultiThreaderBase::GetGlobalDefaultNumberOfThreads(), "number of threads", cmd);
    TCLAP::ValueArg<unsigned int> nbBestVoxel(
        "n","nb-best",
        "Number of voxels selected for response function estimation first step (with best FA values)",
        false, 300, "unsigned int", cmd);

    try
    {
        cmd.parse(argc, argv);
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    using MainFilterType = anima::ODFEstimatorCSDImageFilter<double, double>;
    using DTIFilterType = anima::DTIEstimationImageFilter<double, double>;
    using FAFilterType = anima::DTIScalarMapsImageFilter<3>;
    using InputImageType = MainFilterType::Input3DImageType;

    MainFilterType::Pointer mainFilter = MainFilterType::New();
    mainFilter->SetLambda(lambdaArg.getValue());
    mainFilter->SetTau(tauArg.getValue());
    mainFilter->SetNbBestVoxel(nbBestVoxel.getValue());
    if (orderArg.getValue() % 2 == 0)
        mainFilter->SetLOrder(orderArg.getValue());
    else
        mainFilter->SetLOrder(orderArg.getValue() - 1);
    mainFilter->SetBValueShellSelected(selectedBvalArg.getValue());
    mainFilter->SetNumberOfWorkUnits(nbpArg.getValue());
    int nbPats = anima::setMultipleImageFilterInputsFromFileName<InputImageType, MainFilterType>(inArg.getValue(), mainFilter);

    DTIFilterType::Pointer dtiFilter = DTIFilterType::New();
    for (int i = 0; i < nbPats; i++){
        dtiFilter->SetInput(i, mainFilter->GetInput(i));
    }
    //anima::setMultipleImageFilterInputsFromFileName<InputImageType, DTIFilterType>(inArg.getValue(), dtiFilter);

    using GFReaderType = anima::GradientFileReader<vnl_vector_fixed<double,3>, double>;
    GFReaderType gfReader;
    gfReader.SetGradientFileName(gradArg.getValue());
    gfReader.SetBValueBaseString(bvalArg.getValue());
    gfReader.SetGradientIndependentNormalization(bvalueScaleArg.isSet());
    gfReader.Update();

    //DTI filter needs gradient directions and bvalues as they are in the input files (see animaDTIEstimator implementation)
    GFReaderType::GradientVectorType directions = gfReader.GetGradients();
    GFReaderType::BValueVectorType mb = gfReader.GetBValues();
    for (unsigned int i = 0; i < directions.size(); ++i)
        dtiFilter->AddGradientDirection(i, directions[i]);
    dtiFilter->SetBValuesList(mb);
    dtiFilter->SetNumberOfWorkUnits(nbpArg.getValue());
    dtiFilter->Update();
    mainFilter->SetDtiImage(*dtiFilter->GetOutput());

    FAFilterType::Pointer faFilter = FAFilterType::New();
    faFilter->SetInput(mainFilter->GetDtiImage());
    faFilter->SetNumberOfWorkUnits(nbpArg.getValue());
    faFilter->Update();
    mainFilter->SetFaImage(faFilter->GetFAImage());

    //here, we want to set every gradient with bvalue<=10 to null vector and the associated bvalue to 0
    gfReader.SetB0ValueThreshold(10);
    gfReader.Update();

    directions = gfReader.GetGradients();
    mb = gfReader.GetBValues();
    for (unsigned int i = 0; i < directions.size(); ++i)
        mainFilter->AddGradientDirection(i, directions[i]);
    mainFilter->SetBValuesList(mb);

    itk::TimeProbe tmpTime;
    tmpTime.Start();
    mainFilter->Update();
    tmpTime.Stop();

    std::cout << "\nExecution Time: " << tmpTime.GetTotal() << "s" << std::endl;

    anima::writeImage<MainFilterType::OutputVectorImageType>(resArg.getValue(), mainFilter->GetOutput());

    return EXIT_SUCCESS;
}

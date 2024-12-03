#include <animaGradientFileReader.h>
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
        "-t", "tau",
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

    try
    {
        cmd.parse(argc, argv);
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "Error: " << e.error() << "for argument " << e.argId() << std::endl;
        return EXIT_FAILURE;
    }

    using MainFilterType = anima::ODFEstimatorImageCSDFilter<double, double>;
    using InputImageType = MainFilterType::TInputImage;

    MainFilterType::Pointer mainFilter = MainFilterType::New();
    mainFilter->SetLambda(lambdaArg.getValue());
    mainFilter->SetTau(tauArg.getValue());
    if (orderArg.getValue() % 2 == 0)
        mainFilter->SetLOrder(orderArg.getValue());
    else
        mainFilter->SetLOrder(orderArg.getValue() - 1);

    anima::setMultipleImageFilterInputsFromFileName<InputImageType, MainFilterType>(inArg.getValue(), mainFilter);

    using GFReaderType = anima::GradientFileReader<std::vector<double>, double>;
    GFReaderType gfReader;
    gfReader.SetGradientFileName(gradArg.getValue());
    gfReader.SetBValueBaseString(bvalArg.getValue());
    gfReader.SetGradientIndependentNormalization(bvalueScaleArg.isSet());
    gfReader.SetB0ValueThreshold(10);
    gfReader.Update();

    GFReaderType::GradientVectorType directions = gfReader.GetGradients();
    GFReaderType::BValueVectorType mb = gfReader.GetBValues();

    for (unsigned int i = 0; i < directions.size(); ++i)
        mainFilter->AddGradientDirection(i, directions[i]);

    mainFilter->SetBValuesList(mb);
    mainFilter->SetBValueShellSelected(selectedBvalArg.getValue());
    mainFilter->SetNumberOfWorkUnits(nbpArg.getValue());

    itk::TimeProbe tmpTime;
    tmpTime.Start();
    mainFilter->Update();
    tmpTime.Stop();

    std::cout << "\nExecution Time: " << tmpTime.GetTotal() << "s" << std::endl;

    anima::writeImage<MainFilterType::TOutputImage>(resArg.getValue(), mainFilter->GetOutput());

    return EXIT_SUCCESS;
}

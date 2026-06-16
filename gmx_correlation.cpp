#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "correlation_core.h"

/*! \file
 * \brief GROMACS 2025 trajectory-analysis front end for gmx_correlation.
 *
 * The original tool read trajectories through GROMACS 3/4 C APIs that no
 * longer exist. This file implements the modern GROMACS analysis module,
 * collects one static atom selection over all frames, mean-centers the
 * coordinates, and hands the resulting trajectory to the preserved correlation
 * routines in `correlation_core.cpp` and `kraskov.cpp`.
 */

#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/options.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/cmdlinerunner.h"
#include "gromacs/utility/exceptions.h"

#ifdef GMX_CORRELATION_USE_MPI
#include <mpi.h>
#endif

using namespace gmx;

class Correlation : public TrajectoryAnalysisModule
{
public:
    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    std::string fnMatrix_;
    std::string fnXpm_;
    std::string fnDump_;
    int         skip_    = 1;
    int         k_       = 100;
    bool        fit_     = false;
    bool        linear_  = false;
    bool        inBits_  = false;
    bool        useGpu_   = false;
    int         nthreads_ = 0;
    Selection   sel_;
    /* Frame collection is intentionally done in C++ containers. The conversion
     * to `t_traj` happens only once in finishAnalysis(), where the legacy math
     * code boundary begins. */
    std::vector<std::vector<std::array<double, DIM>>> coords_;
    t_traj      traj_;
    std::string selectionName_;
};

void Correlation::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    /* GROMACS supplies the standard -f, -s, -n, -b, -e, -dt, -pbc, and
     * selection-related options around these module-specific options. */
    static const char* const desc[] = {
        "gmx_correlation computes the mutual-information based correlation of all pairs of atoms in the selected group.",
        "The Kraskov, Stoegbauer and Grassberger nearest-neighbor estimator is used for mutual information, matching the original GROMACS 3/4-era tool.",
        "Use [TT]-linear[tt] for the faster Gaussian mutual-information approximation.",
        "For fitted analysis, pre-fit the trajectory with [TT]gmx trjconv -fit[tt] before running this tool."
    };

    settings->setHelpText(desc);
    /* Requiring topology keeps selection behavior close to the legacy tool,
     * where users chose analysis groups from topology/index information. */
    settings->setFlag(TrajectoryAnalysisSettings::efRequireTop);
    settings->setPBC(true);

    options->addOption(FileNameOption("o")
                               .filetype(OptionFileType::GenericData)
                               .outputFile()
                               .store(&fnMatrix_)
                               .defaultBasename("correl")
                               .required(true)
                               .description("Output correlation matrix"));
    options->addOption(FileNameOption("m")
                               .filetype(OptionFileType::GenericData)
                               .outputFile()
                               .store(&fnXpm_)
                               .defaultBasename("correl")
                               .description("Optional XPM matrix output"));
    options->addOption(SelectionOption("select")
                               .store(&sel_)
                               .required()
                               .onlyAtoms()
                               .description("Atoms to include in the correlation matrix"));
    options->addOption(IntegerOption("skip").store(&skip_).description("Only use every nr-th frame"));
    options->addOption(IntegerOption("k").store(&k_).description("Use k-nearest neighbours for estimation"));
    options->addOption(BooleanOption("fit").store(&fit_).description("Compatibility option; pre-fit trajectory instead"));
    options->addOption(BooleanOption("linear").store(&linear_).description("Compute Gaussian linearized mutual information"));
    options->addOption(BooleanOption("mi").store(&inBits_).description("Output mutual information instead of coefficient"));
    options->addOption(BooleanOption("gpu").store(&useGpu_).description("Use GPU (CUDA) for Kraskov MI; falls back to CPU if unavailable"));
    options->addOption(IntegerOption("nt").store(&nthreads_).description("Number of CPU threads for Kraskov (0 = all available)"));
    options->addOption(FileNameOption("dump")
                               .filetype(OptionFileType::GenericData)
                               .outputFile()
                               .store(&fnDump_)
                               .defaultBasename("traj_dump")
                               .description("Dump mean-centered trajectory for comparison with legacy tool"));
}

void Correlation::initAnalysis(const TrajectoryAnalysisSettings& /*settings*/, const TopologyInformation& /*top*/)
{
    if (skip_ < 1)
    {
        GMX_THROW(InconsistentInputError("The -skip value must be at least 1."));
    }
    if (k_ < 1)
    {
        GMX_THROW(InconsistentInputError("The -k value must be at least 1."));
    }
    if (fit_)
    {
        /* The old fitting path depended on removed routines such as
         * read_tps_conf(), rm_pbc(), reset_x(), and do_fit(). Keeping -fit as a
         * rejected compatibility option gives users a clear migration path
         * instead of silently producing unfitted results. */
        GMX_THROW(NotImplementedError("The GROMACS 2025 port does not perform fitting internally. Pre-fit the trajectory with gmx trjconv -fit and rerun without -fit."));
    }
    selectionName_ = sel_.name();
}

void Correlation::analyzeFrame(int frnr, const t_trxframe& /*fr*/, t_pbc* /*pbc*/, TrajectoryAnalysisModuleData* /*pdata*/)
{
    if (frnr % skip_ != 0)
    {
        return;
    }

    const Selection& sel = TrajectoryAnalysisModuleData::parallelSelection(sel_);
    if (coords_.empty())
    {
        coords_.resize(sel.posCount());
    }
    if (sel.posCount() != static_cast<int>(coords_.size()))
    {
        /* The correlation matrix has a fixed atom-pair interpretation, so
         * dynamic selections that change size across frames are not valid. */
        GMX_THROW(InconsistentInputError("The selected atom count changed between frames; use a static atom selection."));
    }

    for (int i = 0; i < sel.posCount(); ++i)
    {
        const auto& x = sel.position(i).x();
        coords_[i].push_back({ x[XX], x[YY], x[ZZ] });
    }
}

void Correlation::finishAnalysis(int /*nframes*/)
{
    if (coords_.empty() || coords_[0].empty())
    {
        GMX_THROW(InconsistentInputError("No trajectory frames were selected for analysis."));
    }
    if (k_ >= static_cast<int>(coords_[0].size()))
    {
        GMX_THROW(InconsistentInputError("The -k value must be smaller than the number of analyzed frames."));
    }

    traj_.natoms  = static_cast<int>(coords_.size());
    traj_.nframes = static_cast<int>(coords_[0].size());
    snew(traj_.x, traj_.natoms);
    snew(traj_.xav, traj_.natoms);

    for (int atom = 0; atom < traj_.natoms; ++atom)
    {
        /* Match the legacy preprocessing: store fluctuations around each
         * atom's average position before computing mutual information. */
        snew(traj_.x[atom], traj_.nframes);
        for (int frame = 0; frame < traj_.nframes; ++frame)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                traj_.xav[atom][dim] += coords_[atom][frame][dim];
            }
        }
        for (int dim = 0; dim < DIM; ++dim)
        {
            traj_.xav[atom][dim] /= traj_.nframes;
        }
        for (int frame = 0; frame < traj_.nframes; ++frame)
        {
            for (int dim = 0; dim < DIM; ++dim)
            {
                traj_.x[atom][frame][dim] = coords_[atom][frame][dim] - traj_.xav[atom][dim];
            }
        }
    }

    if (!fnDump_.empty())
    {
        FILE* dp = std::fopen(fnDump_.c_str(), "w");
        if (!dp)
        {
            GMX_THROW(FileIOError("Cannot open dump file: " + fnDump_));
        }
        std::fprintf(dp, "%d %d\n", traj_.natoms, traj_.nframes);
        for (int frame = 0; frame < traj_.nframes; ++frame)
        {
            for (int atom = 0; atom < traj_.natoms; ++atom)
            {
                std::fprintf(dp, "%.10g %.10g %.10g\n",
                             traj_.x[atom][frame][XX],
                             traj_.x[atom][frame][YY],
                             traj_.x[atom][frame][ZZ]);
            }
        }
        std::fclose(dp);
    }
}

void Correlation::writeOutput()
{
    const int natoms  = traj_.natoms;
    const int nframes = traj_.nframes;
    std::vector<double> result(natoms * natoms, 1.0);

    using Clock = std::chrono::steady_clock;

    if (linear_)
    {
        fprintf(stderr, "\nComputing Gaussian correlation matrix "
                "(%d atoms, %d frames)...\n", natoms, nframes);
        const auto t0 = Clock::now();
        gauss_corrmatrix(&traj_, result.data());
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - t0).count();
        fprintf(stderr, "Gaussian matrix done in %.2f s\n", elapsed);
    }
    else
    {
        if (useGpu_ && !gpu_available()) {
            fprintf(stderr, "Note: --gpu requested but no GPU available — running on CPU.\n");
            useGpu_ = false;
        }
        const char* backend = useGpu_ ? "GPU" : "CPU";
        fprintf(stderr, "\nComputing Kraskov correlation matrix "
                "(%d atoms, %d frames, k=%d) on %s...\n",
                natoms, nframes, k_, backend);
        const auto t0 = Clock::now();
        kraskov_corrmatrix(&traj_, result.data(), k_, useGpu_, nthreads_);
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - t0).count();
        fprintf(stderr, "Kraskov matrix done in %.2f s\n", elapsed);
    }

    if (!inBits_)
    {
        pearsify(result.data(), natoms, DIM);
    }

    fprintf(stderr, "Writing output to %s\n", fnMatrix_.c_str());
    write_matrix(result.data(), natoms, fnMatrix_.c_str());
    if (!fnXpm_.empty())
    {
        const std::string title  = selectionName_ + " correlation matrix";
        const char*       legend = inBits_ ? "MI (bits)" : "r(MI)";
        write_xpm_matrix(result.data(), natoms, fnXpm_.c_str(), title.c_str(), legend);
        fprintf(stderr, "Writing XPM to %s\n", fnXpm_.c_str());
    }

    done_traj(&traj_);
}

int main(int argc, char* argv[])
{
#ifdef GMX_CORRELATION_USE_MPI
    /* GROMACS' command-line runner is not responsible for MPI_Init() for this
     * standalone tool, so MPI builds initialize/finalize around it. */
    int mpiWasInitialized = 0;
    MPI_Initialized(&mpiWasInitialized);
    if (!mpiWasInitialized)
    {
        MPI_Init(&argc, &argv);
    }
#endif

    const int rc = TrajectoryAnalysisCommandLineRunner::runAsMain<Correlation>(argc, argv);

#ifdef GMX_CORRELATION_USE_MPI
    if (!mpiWasInitialized)
    {
        MPI_Finalize();
    }
#endif
    return rc;
}

/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef _DIST_KMEANS_INIT_INCLUDED_
#define _DIST_KMEANS_INIT_INCLUDED_

#include "dist_custom.h"
#include "map_reduce_star.h"

namespace dist_custom
{

// unsupported methods
template <typename fptype, daal::algorithms::kmeans::init::Method method>
class dist_custom<kmeans_init_manager<fptype, method> >
{
public:
    typedef kmeans_init_manager<fptype, method> Algo;

    template <typename... Ts>
    static typename Algo::iomb_type::result_type compute(Algo & algo, const Ts &... inputs)
    {
        std::cerr << "kmeans-init: selected method not supported yet; returning empty centroids.\n";
        return typename Algo::iomb_type::result_type();
    }
};

// oneDAL kmeans_init Distributed algos do not return a proper result (like batch), we need to create one
template <typename fptype, daal::algorithms::kmeans::init::Method method>
typename kmeans_init_manager<fptype, method>::iomb_type::result_type mk_kmi_result(const daal::data_management::NumericTablePtr & centroids)
{
    typename kmeans_init_manager<fptype, method>::iomb_type::result_type res(new typename kmeans_init_manager<fptype, method>::iomb_type::result_type::ElementType);
    res->set(daal::algorithms::kmeans::init::centroids, daal::data_management::convertToHomogen<fptype>(*centroids.get()));
    return res;
}

// kmi using simple map_reduce_star (random and deterministic)
template <typename fptype, daal::algorithms::kmeans::init::Method method>
class kmi_map_reduce
{
public:
    typedef kmeans_init_manager<fptype, method> Algo;

    typename Algo::iomb_type::result_type static map_reduce(Algo & algo, const daal::data_management::NumericTablePtr input)
    {
        auto tcvr = get_transceiver();

        size_t tot_rows  = input->getNumberOfRows();
        size_t start_row = tot_rows;
        // first determine total number of rows
        tcvr->reduce_all(&tot_rows, 1, transceiver_iface::OP_SUM);
        // determine start of my chunk
        tcvr->reduce_exscan(&start_row, 1, transceiver_iface::OP_SUM);
        if (tcvr->me() == 0) start_row = 0;

        auto res = map_reduce_star::map_reduce_star<Algo>::map_reduce(algo, input, tot_rows, start_row);

        return mk_kmi_result<fptype, method>(res);
    }

    template <typename... Ts>
    static typename Algo::iomb_type::result_type compute(Algo & algo, const Ts &... inputs)
    {
        return map_reduce(algo, get_table(inputs)...);
    }
};

// specialize dist_custom for kemans_init<randomDense>
template <typename fptype>
class dist_custom<kmeans_init_manager<fptype, daal::algorithms::kmeans::init::randomDense> > : public kmi_map_reduce<fptype, daal::algorithms::kmeans::init::randomDense>
{};

// specialize dist_custom for kemans_init<deterministicDense>
template <typename fptype>
class dist_custom<kmeans_init_manager<fptype, daal::algorithms::kmeans::init::deterministicDense> > : public kmi_map_reduce<fptype, daal::algorithms::kmeans::init::deterministicDense>
{};

// specialize dist_custom for kemans_init<plusPlusDense>
template <typename fptype>
class dist_custom<kmeans_init_manager<fptype, daal::algorithms::kmeans::init::plusPlusDense> >
{
public:
    typedef kmeans_init_manager<fptype, daal::algorithms::kmeans::init::plusPlusDense> Algo;
    /*
          step1 is a pre-processing step done before the iteration.
          The iteration identifieds one centroids after the other.
          step4 is the equivalent of step1, but within the iteration.
          step4/1 are meaningful only on a single rank:
              step1 produces output only on one rank
              step4 is executed only on one rank, the output of step3 determines wich rank does it
          step2 processes output of step1/4 on each node, so output of step1/4 need bcasting
          step2 results get gathered on root and processed in step3
          step3 result is sent to one rank for executing step4
         */
    static typename Algo::iomb_type::result_type kmi(Algo & algo, const daal::data_management::NumericTablePtr input)
    {
        auto tcvr  = get_transceiver();
        int rank   = tcvr->me();
        int nRanks = tcvr->nMembers();

        // first determine total number of rows
        size_t tot_rows  = input->getNumberOfRows();
        size_t start_row = tot_rows;
        // first determine total number of rows
        tcvr->reduce_all(&tot_rows, 1, transceiver_iface::OP_SUM);
        // determine start of my chunk
        tcvr->reduce_exscan(&start_row, 1, transceiver_iface::OP_SUM);
        if (rank == 0) start_row = 0;

        /* Internal data to be stored on the local nodes */
        daal::data_management::DataCollectionPtr localNodeData;
        /* Numeric table to collect the results */
        daal::data_management::RowMergedNumericTablePtr pCentroids(new daal::data_management::RowMergedNumericTable());
        // First step on each rank (output var will be used for output of step4 as well)
        auto step14Out = algo.run_step1Local(input, tot_rows, start_row)->get(daal::algorithms::kmeans::init::partialCentroids);
        // Only one rank actually computes centroids, we need to identify rank and bcast centroids to all others
        int data_rank = not_empty(step14Out) ? rank : -1;
        tcvr->reduce_all(&data_rank, 1, transceiver_iface::OP_MAX);
        tcvr->bcast(step14Out, data_rank);
        auto step2In = step14Out;

        pCentroids->addNumericTable(step2In);

        for (size_t iCenter = 1; iCenter < algo._nClusters; ++iCenter)
        {
            // run step2 on each rank
            auto s2res = algo.run_step2Local(input, localNodeData, step2In, false);
            if (iCenter == 1) localNodeData = s2res->get(daal::algorithms::kmeans::init::internalResult);
            auto s2Out = s2res->get(daal::algorithms::kmeans::init::outputOfStep2ForStep3);
            //  and gather result on root
            auto s3In        = tcvr->gather(s2Out);
            const int S34TAG = 3003;
            // The input for s4 will be stored in s4In
            daal::data_management::NumericTablePtr s4In;
            // run step3 on root
            // step3's output provides input for only one rank, this rank needs to be identified to run step4
            if (rank == 0)
            {
                auto step3Output = algo.run_step3Master(s3In);
                for (int i = 0; i < nRanks; ++i)
                {
                    if (step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep4, i))
                    {
                        data_rank = i;
                        break;
                    }
                }
                tcvr->bcast(data_rank, 0);
                if (data_rank)
                {
                    tcvr->send(step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep4, data_rank), data_rank, S34TAG);
                }
                else
                {
                    s4In = step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep4, 0);
                }
            }
            else
            { // non-roots get notified about who will do step 4 with output from step3
                tcvr->bcast(data_rank, 0);
                if (rank == data_rank)
                {
                    s4In = tcvr->recv<daal::data_management::NumericTablePtr>(0, S34TAG);
                }
            }
            // only one rank actually executes step4
            if (rank == data_rank)
            {
                // run step4 on responsible rank, result will feed into step2 of next iteration
                step14Out = algo.run_step4Local(input, localNodeData, s4In);
            }
            // similar to output of step1, output of step4 gets bcasted to all ranks and fed into step2 of next iteration
            tcvr->bcast(step14Out, data_rank);
            step2In = step14Out;
            pCentroids->addNumericTable(step2In);
        }

        // Now create result object, set centroids and return
        return mk_kmi_result<fptype, daal::algorithms::kmeans::init::plusPlusDense>(pCentroids);
    }

    template <typename... Ts>
    static typename Algo::iomb_type::result_type compute(Algo & algo, Ts &... inputs)
    {
        return kmi(algo, get_table(inputs)...);
    }
};

// specialize dist_custom for kemans_init<parallelPlusDense>
template <typename fptype>
class dist_custom<kmeans_init_manager<fptype, daal::algorithms::kmeans::init::parallelPlusDense> >
{
public:
    typedef kmeans_init_manager<fptype, daal::algorithms::kmeans::init::parallelPlusDense> Algo;
    /*
            step1 provides initial input for step2, inside the loop step4 produces the input for step2.
            We have to keep input for step2 because it will also be used as input for final step5.
            Now we iterate/loop for nRounds:
                - step2 computes on each rank
                - we gather results from all step2's
                - gathered data from step2 is input to step3, executed on root only
                - output of step3 is scattered to all ranks
                - ranks which received non-empty output from step3 will execute step4 on its data
                - results of step4 are input for step2 of next iteration
            After the loop
                - we execute step2 one more time with data from last iteration of loop on each rank
                - and finally select the initial centroids in step5 on root
            The resulting centroids are broadcasted to all processes.
        */
    static typename Algo::iomb_type::result_type kmi(Algo & algo, const daal::data_management::NumericTablePtr input)
    {
        auto tcvr  = get_transceiver();
        int rank   = tcvr->me();
        int nRanks = tcvr->nMembers();

        // first determine total number of rows
        size_t tot_rows  = input->getNumberOfRows();
        size_t start_row = tot_rows;
        // first determine total number of rows
        tcvr->reduce_all(&tot_rows, 1, transceiver_iface::OP_SUM);
        // determine start of my chunk
        tcvr->reduce_exscan(&start_row, 1, transceiver_iface::OP_SUM);
        if (rank == 0) start_row = 0;

        // Internal data to be stored on the local nodes
        daal::data_management::DataCollectionPtr localNodeData;
        // First step on each rank (output var will be used for output of step4 as well)
        auto step14Out = algo.run_step1Local(input, tot_rows, start_row)->get(daal::algorithms::kmeans::init::partialCentroids);
        // Only one rank actually computes centroids, we need to identify rank and bcast centroids to all others
        int data_rank = not_empty(step14Out) ? rank : -1;
        tcvr->reduce_all(&data_rank, 1, transceiver_iface::OP_MAX);
        tcvr->bcast(step14Out, data_rank);
        auto step2In = step14Out;

        // default value of nRounds used by all steps
        const size_t nRounds = daal::algorithms::kmeans::init::Parameter(algo._nClusters).nRounds;

        // vector with results of step2 for input into step5
        std::vector<daal::data_management::NumericTablePtr> s2InForStep5;
        if (rank == 0)
        {
            s2InForStep5.push_back(step2In);
        }

        // Here we will store the output of step3 for step5
        daal::services::interface1::SharedPtr<daal::data_management::interface1::SerializationIface> outputOfStep3ForStep5;

        for (size_t iRound = 0; iRound < nRounds; ++iRound)
        {
            // run step2 on each rank
            auto s2res = algo.run_step2Local(input, localNodeData, step2In, false);
            if (iRound == 0) localNodeData = s2res->get(daal::algorithms::kmeans::init::internalResult);
            auto s2Out = s2res->get(daal::algorithms::kmeans::init::outputOfStep2ForStep3);
            //  and gather result on root
            auto s3In = tcvr->gather(s2Out);

            const int S34TAG = 3003;
            // The input for s4 will be stored in s4In
            daal::data_management::NumericTablePtr s4In;
            // run step3 on root and send results to non-roots
            if (rank == 0)
            {
                auto step3Output = algo.run_step3Master(s3In);
                // output of step3 in the last iteration will be used in step5
                if (iRound == nRounds - 1)
                {
                    outputOfStep3ForStep5 = step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep5);
                }
                s4In = step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep4, 0);
                for (int i = 1; i < nRanks; i++)
                {
                    tcvr->send(step3Output->get(daal::algorithms::kmeans::init::outputOfStep3ForStep4, i), i, S34TAG); // it can be NULL
                }
            }
            else
            { // non-roots get messages with output from step3
                s4In = tcvr->recv<daal::data_management::NumericTablePtr>(0, S34TAG);
            }
            // if we have a data for step4 then run it
            if (s4In)
            {
                step14Out = algo.run_step4Local(input, localNodeData, s4In);
            }
            else
            {
                step14Out = daal::data_management::NumericTablePtr();
            }

            // we need to gather all exist results on root, merge them into one table and then share it with all non-roots
            auto step14OutMaster = tcvr->gather(step14Out);
            daal::data_management::RowMergedNumericTablePtr step4OutMerged(new daal::data_management::RowMergedNumericTable());
            if (rank == 0)
            {
                for (int i = 0; i < step14OutMaster.size(); i++)
                {
                    // we expect that some of results can be NULL
                    if (step14OutMaster[i])
                    {
                        step4OutMerged->addNumericTable(step14OutMaster[i]);
                    }
                }
            }
            tcvr->bcast(step4OutMerged, 0);
            step2In = daal::data_management::convertToHomogen<fptype>(*step4OutMerged.get());

            // we add results of each iteration to input of step5
            if (rank == 0)
            {
                s2InForStep5.push_back(step2In);
            }
        }

        // One more step 2
        auto s2ResForStep5 = algo.run_step2Local(input, localNodeData, step2In, true);
        auto s2OutForStep5 = s2ResForStep5->get(daal::algorithms::kmeans::init::outputOfStep2ForStep5);
        auto s5In          = tcvr->gather(s2OutForStep5);
        daal::data_management::NumericTablePtr s5Res;
        if (rank == 0)
        {
            s5Res = algo.run_step5Master(s2InForStep5, s5In, outputOfStep3ForStep5);
        }
        tcvr->bcast(s5Res, 0);
        return mk_kmi_result<fptype, daal::algorithms::kmeans::init::parallelPlusDense>(s5Res);
    }

    template <typename... Ts>
    static typename Algo::iomb_type::result_type compute(Algo & algo, Ts &... inputs)
    {
        return kmi(algo, get_table(inputs)...);
    }
};
} // namespace dist_custom

#endif // _DIST_KMEANS_INIT_INCLUDED_

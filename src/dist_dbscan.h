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

#ifndef _DIST_DBSCAN_INCLUDED_
#define _DIST_DBSCAN_INCLUDED_

#include "dist_custom.h"
#include "daal4py_defines.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

namespace dist_custom
{

template <typename T1, typename T2>
class dist
{
public:
    typedef std::vector<daal::byte> ByteBuffer;

    size_t serializeDAALObject(SerializationIface * pData, ByteBuffer & buffer)
    {
        /* Create a data archive to serialize the numeric table */
        InputDataArchive dataArch;

        /* Serialize the numeric table into the data archive */
        pData->serialize(dataArch);

        /* Get the length of the serialized data in bytes */
        const size_t length = dataArch.getSizeOfArchive();

        /* Store the serialized data in an array */
        buffer.resize(length);
        if (length) dataArch.copyArchiveToArray(&buffer[0], length);
        return length;
    }

    SerializationIfacePtr deserializeDAALObject(daal::byte * buff, size_t length)
    {
        /* Create a data archive to deserialize the object */
        OutputDataArchive dataArch(buff, length);

        /* Deserialize the numeric table from the data archive */
        return dataArch.getAsSharedPtr();
    }

    typedef T2 algorithmFPType; /* Algorithm floating-point type */

    /* Algorithm parameters */
    T2 epsilon;
    size_t minObservations;

    NumericTablePtr dataTable;

    DataCollectionPtr partitionedData;
    DataCollectionPtr partitionedPartialOrders;

    DataCollectionPtr partialSplits;
    DataCollectionPtr partialBoundingBoxes;

    DataCollectionPtr haloData;
    DataCollectionPtr haloDataIndices;
    DataCollectionPtr haloBlocks;

    DataCollectionPtr queries;

    DataCollectionPtr assignmentQueries;

    NumericTablePtr clusterStructure;
    NumericTablePtr finishedFlag;
    NumericTablePtr nClusters;
    NumericTablePtr clusterOffset;
    NumericTablePtr assignments;
    NumericTablePtr totalNClusters;

    int rankId, comm_size;

    const int step2ResultBoundingBoxTag                = 1;
    const int step3ResultSplitTag                      = 2;
    const int step4ResultPartitionedDataTag            = 3;
    const int step4ResultPartitionedPartialOrdersTag   = 4;
    const int step5ResultPartitionedHaloDataTag        = 5;
    const int step5ResultPartitionedHaloDataIndicesTag = 6;
    const int step5ResultPartitionedHaloBlocksTag      = 7;
    const int step6ResultQueriesTag                    = 8;
    const int step8ResultQueriesTag                    = 9;
    const int step8ResultNClustersTag                  = 10;
    const int resultFinishedFlagTag                    = 11;
    const int step7ResultFinishedFlagTag               = 12;
    const int step9ResultNClustersTag                  = 13;
    const int step9ResultClusterOffsetsTag             = 14;
    const int step10ResultQueriesTag                   = 15;
    const int step11ResultQueriesTag                   = 16;
    const int step12ResultAssignmentQueriesTag         = 17;

    transceiver * tcvr;

    int main(const T1 & input1)
    {
        tcvr      = get_transceiver();
        rankId    = tcvr->me();
        comm_size = tcvr->nMembers();

        dataTable = input1;

        geometricPartitioning();

        clustering();

        return 0;
    }

    void geometricPartitioning()
    {
        dbscan::Distributed<step1Local, algorithmFPType, dbscan::defaultDense> step1(rankId, comm_size);
        step1.input.set(dbscan::step1Data, dataTable);
        step1.compute();

        partitionedData          = DataCollectionPtr(new DataCollection());
        partitionedPartialOrders = DataCollectionPtr(new DataCollection());

        partitionedData->push_back(dataTable);
        partitionedPartialOrders->push_back(step1.getPartialResult()->get(dbscan::partialOrder));

        size_t beginId = 0;
        size_t endId   = comm_size;

        while (true)
        {
            const size_t curNPartitions = endId - beginId;
            if (curNPartitions == 1)
            {
                break;
            }

            partialSplits        = DataCollectionPtr(new DataCollection());
            partialBoundingBoxes = DataCollectionPtr(new DataCollection());

            dbscan::Distributed<step2Local, algorithmFPType, dbscan::defaultDense> step2(rankId - beginId, curNPartitions);
            step2.input.set(dbscan::partialData, partitionedData);
            step2.compute();
            NumericTablePtr curBoundingBox = step2.getPartialResult()->get(dbscan::boundingBox);

            sendTableAllToAll(beginId, endId, rankId, step2ResultBoundingBoxTag, curBoundingBox, partialBoundingBoxes);

            const size_t leftPartitions  = curNPartitions / 2;
            const size_t rightPartitions = curNPartitions - leftPartitions;

            dbscan::Distributed<step3Local, algorithmFPType, dbscan::defaultDense> step3(leftPartitions, rightPartitions);
            step3.input.set(dbscan::partialData, partitionedData);
            step3.input.set(dbscan::step3PartialBoundingBoxes, partialBoundingBoxes);
            step3.compute();
            NumericTablePtr curSplit = step3.getPartialResult()->get(dbscan::split);

            sendTableAllToAll(beginId, endId, rankId, step3ResultSplitTag, curSplit, partialSplits);

            dbscan::Distributed<step4Local, algorithmFPType, dbscan::defaultDense> step4(leftPartitions, rightPartitions);
            step4.input.set(dbscan::partialData, partitionedData);
            step4.input.set(dbscan::step4PartialOrders, partitionedPartialOrders);
            step4.input.set(dbscan::step4PartialSplits, partialSplits);
            step4.compute();

            DataCollectionPtr curPartitionedData          = step4.getPartialResult()->get(dbscan::partitionedData);
            DataCollectionPtr curPartitionedPartialOrders = step4.getPartialResult()->get(dbscan::partitionedPartialOrders);

            partitionedData          = DataCollectionPtr(new DataCollection());
            partitionedPartialOrders = DataCollectionPtr(new DataCollection());

            sendCollectionAllToAll(beginId, endId, rankId, step4ResultPartitionedDataTag, curPartitionedData, partitionedData);
            sendCollectionAllToAll(beginId, endId, rankId, step4ResultPartitionedPartialOrdersTag, curPartitionedPartialOrders, partitionedPartialOrders);

            if (rankId < beginId + leftPartitions)
            {
                endId = beginId + leftPartitions;
            }
            else
            {
                beginId = beginId + leftPartitions;
            }
        }
    }

    void clustering()
    {
        partialBoundingBoxes = DataCollectionPtr(new DataCollection());
        haloData             = DataCollectionPtr(new DataCollection());
        haloDataIndices      = DataCollectionPtr(new DataCollection());
        haloBlocks           = DataCollectionPtr(new DataCollection());

        dbscan::Distributed<step2Local, algorithmFPType, dbscan::defaultDense> step2(rankId, comm_size);
        step2.input.set(dbscan::partialData, partitionedData);
        step2.compute();
        NumericTablePtr curBoundingBox = step2.getPartialResult()->get(dbscan::boundingBox);

        sendTableAllToAll(0, comm_size, rankId, step2ResultBoundingBoxTag, curBoundingBox, partialBoundingBoxes, true /* preserveOrder */);

        dbscan::Distributed<step5Local, algorithmFPType, dbscan::defaultDense> step5(rankId, comm_size, epsilon);
        step5.input.set(dbscan::partialData, partitionedData);
        step5.input.set(dbscan::step5PartialBoundingBoxes, partialBoundingBoxes);
        step5.compute();
        DataCollectionPtr curHaloData        = step5.getPartialResult()->get(dbscan::partitionedHaloData);
        DataCollectionPtr curHaloDataIndices = step5.getPartialResult()->get(dbscan::partitionedHaloDataIndices);
        DataCollectionPtr curHaloBlocks(new DataCollection());

        for (size_t destId = 0; destId < curHaloData->size(); destId++)
        {
            NumericTablePtr dataTable = services::staticPointerCast<NumericTable, SerializationIface>((*curHaloData)[destId]);
            if (dataTable->getNumberOfRows() > 0)
            {
                curHaloBlocks->push_back(HomogenNumericTable<int>::create(1, 1, NumericTableIface::doAllocate, static_cast<int>(rankId)));
            }
            else
            {
                curHaloBlocks->push_back(NumericTablePtr());
            }
        }

        sendCollectionAllToAll(0, comm_size, rankId, step5ResultPartitionedHaloDataTag, curHaloData, haloData);
        sendCollectionAllToAll(0, comm_size, rankId, step5ResultPartitionedHaloDataIndicesTag, curHaloDataIndices, haloDataIndices);
        sendCollectionAllToAll(0, comm_size, rankId, step5ResultPartitionedHaloBlocksTag, curHaloBlocks, haloBlocks);

        queries = DataCollectionPtr(new DataCollection());

        dbscan::Distributed<step6Local, algorithmFPType, dbscan::defaultDense> step6(rankId, comm_size, epsilon, minObservations);

        step6.input.set(dbscan::partialData, partitionedData);
        step6.input.set(dbscan::haloData, haloData);
        step6.input.set(dbscan::haloDataIndices, haloDataIndices);
        step6.input.set(dbscan::haloBlocks, haloBlocks);
        step6.compute();
        clusterStructure = step6.getPartialResult()->get(dbscan::step6ClusterStructure);
        finishedFlag     = step6.getPartialResult()->get(dbscan::step6FinishedFlag);
        nClusters        = step6.getPartialResult()->get(dbscan::step6NClusters);

        DataCollectionPtr curQueries = step6.getPartialResult()->get(dbscan::step6Queries);

        sendCollectionAllToAll(0, comm_size, rankId, step6ResultQueriesTag, curQueries, queries);

        while (computeFinishedFlag() == 0)
        {
            dbscan::Distributed<step8Local, algorithmFPType, dbscan::defaultDense> step8(rankId, comm_size);
            step8.input.set(dbscan::step8InputClusterStructure, clusterStructure);
            step8.input.set(dbscan::step8InputNClusters, nClusters);
            step8.input.set(dbscan::step8PartialQueries, queries);
            step8.compute();

            clusterStructure = step8.getPartialResult()->get(dbscan::step8ClusterStructure);
            finishedFlag     = step8.getPartialResult()->get(dbscan::step8FinishedFlag);
            nClusters        = step8.getPartialResult()->get(dbscan::step8NClusters);

            DataCollectionPtr curQueries = step8.getPartialResult()->get(dbscan::step8Queries);

            queries = DataCollectionPtr(new DataCollection());

            sendCollectionAllToAll(0, comm_size, rankId, step8ResultQueriesTag, curQueries, queries);
        }

        if (rankId == 0)
        {
            DataCollectionPtr partialNClusters(new DataCollection());
            sendTableAllToMaster(0, comm_size, rankId, step8ResultNClustersTag, nClusters, partialNClusters);

            dbscan::Distributed<step9Master, algorithmFPType, dbscan::defaultDense> step9;
            step9.input.set(dbscan::partialNClusters, partialNClusters);
            step9.compute();
            step9.finalizeCompute();

            totalNClusters = step9.getResult()->get(dbscan::step9NClusters);
            sendTableMasterToAll(0, comm_size, rankId, step9ResultNClustersTag, totalNClusters, totalNClusters);

            DataCollectionPtr curClusterOffsets = step9.getPartialResult()->get(dbscan::clusterOffsets);
            sendCollectionMasterToAll(0, comm_size, rankId, step9ResultClusterOffsetsTag, curClusterOffsets, clusterOffset);
        }
        else
        {
            DataCollectionPtr partialNClusters;
            sendTableAllToMaster(0, comm_size, rankId, step8ResultNClustersTag, nClusters, partialNClusters);

            sendTableMasterToAll(0, comm_size, rankId, step9ResultNClustersTag, totalNClusters, totalNClusters);

            DataCollectionPtr curClusterOffsets;
            sendCollectionMasterToAll(0, comm_size, rankId, step9ResultClusterOffsetsTag, curClusterOffsets, clusterOffset);
        }

        queries = DataCollectionPtr(new DataCollection());

        dbscan::Distributed<step10Local, algorithmFPType, dbscan::defaultDense> step10(rankId, comm_size);
        step10.input.set(dbscan::step10InputClusterStructure, clusterStructure);
        step10.input.set(dbscan::step10ClusterOffset, clusterOffset);
        step10.compute();

        clusterStructure = step10.getPartialResult()->get(dbscan::step10ClusterStructure);
        finishedFlag     = step10.getPartialResult()->get(dbscan::step10FinishedFlag);

        curQueries = step10.getPartialResult()->get(dbscan::step10Queries);

        sendCollectionAllToAll(0, comm_size, rankId, step10ResultQueriesTag, curQueries, queries);

        while (computeFinishedFlag() == 0)
        {
            dbscan::Distributed<step11Local, algorithmFPType, dbscan::defaultDense> step11(rankId, comm_size);
            step11.input.set(dbscan::step11InputClusterStructure, clusterStructure);
            step11.input.set(dbscan::step11PartialQueries, queries);
            step11.compute();

            clusterStructure = step11.getPartialResult()->get(dbscan::step11ClusterStructure);
            finishedFlag     = step11.getPartialResult()->get(dbscan::step11FinishedFlag);

            DataCollectionPtr curQueries = step11.getPartialResult()->get(dbscan::step11Queries);

            queries = DataCollectionPtr(new DataCollection());
            sendCollectionAllToAll(0, comm_size, rankId, step11ResultQueriesTag, curQueries, queries);
        }

        assignmentQueries = DataCollectionPtr(new DataCollection());

        dbscan::Distributed<step12Local, algorithmFPType, dbscan::defaultDense> step12(rankId, comm_size);
        step12.input.set(dbscan::step12InputClusterStructure, clusterStructure);
        step12.input.set(dbscan::step12PartialOrders, partitionedPartialOrders);
        step12.compute();

        DataCollectionPtr curAssignmentQueries = step12.getPartialResult()->get(dbscan::assignmentQueries);

        sendCollectionAllToAll(0, comm_size, rankId, step12ResultAssignmentQueriesTag, curAssignmentQueries, assignmentQueries);

        dbscan::Distributed<step13Local, algorithmFPType, dbscan::defaultDense> step13;
        step13.input.set(dbscan::partialAssignmentQueries, assignmentQueries);
        step13.compute();
        step13.finalizeCompute();

        assignments = step13.getResult()->get(dbscan::step13Assignments);
    }

    int computeFinishedFlag()
    {
        if (rankId == 0)
        {
            DataCollectionPtr partialFinishedFlags(new DataCollection());
            sendTableAllToMaster(0, comm_size, rankId, resultFinishedFlagTag, finishedFlag, partialFinishedFlags);

            dbscan::Distributed<step7Master, algorithmFPType, dbscan::defaultDense> step7;
            step7.input.set(dbscan::partialFinishedFlags, partialFinishedFlags);
            step7.compute();
            finishedFlag = step7.getPartialResult()->get(dbscan::finishedFlag);

            sendTableMasterToAll(0, comm_size, rankId, step7ResultFinishedFlagTag, finishedFlag, finishedFlag);

            int finishedFlagValue = finishedFlag->getValue<int>(0, 0);
            return finishedFlagValue;
        }
        else
        {
            DataCollectionPtr partialFinishedFlags;
            sendTableAllToMaster(0, comm_size, rankId, resultFinishedFlagTag, finishedFlag, partialFinishedFlags);

            sendTableMasterToAll(0, comm_size, rankId, step7ResultFinishedFlagTag, finishedFlag, finishedFlag);

            int finishedFlagValue = finishedFlag->getValue<int>(0, 0);
            return finishedFlagValue;
        }
    }

    void sendCollectionAllToAll(size_t beginId, size_t endId, size_t curId, int tag, DataCollectionPtr & collection, DataCollectionPtr & destCollection)
    {
        size_t nIds    = endId - beginId;
        size_t nShifts = 1;
        while (nShifts < nIds) nShifts <<= 1;

        for (size_t shift = 0; shift < nShifts; shift++)
        {
            size_t partnerId = ((curId - beginId) ^ shift) + beginId;
            if (partnerId < beginId || partnerId >= endId)
            {
                continue;
            }

            NumericTablePtr table = NumericTable::cast((*collection)[partnerId - beginId]);
            NumericTablePtr partnerTable;

            if (partnerId == curId)
            {
                partnerTable = table;
            }
            else
            {
                if (curId < partnerId)
                {
                    sendTable(table, partnerId, tag);
                    recvTable(partnerTable, partnerId, tag);
                }
                else
                {
                    recvTable(partnerTable, partnerId, tag);
                    sendTable(table, partnerId, tag);
                }
            }

            if (partnerTable.get() && partnerTable->getNumberOfRows() > 0)
            {
                destCollection->push_back(partnerTable);
            }
        }
    }

    void sendTableAllToAll(size_t beginId, size_t endId, size_t curId, int tag, NumericTablePtr & table, DataCollectionPtr & destCollection, bool preserveOrder = false)
    {
        size_t nIds    = endId - beginId;
        size_t nShifts = 1;
        while (nShifts < nIds) nShifts <<= 1;

        if (preserveOrder)
        {
            destCollection = DataCollectionPtr(new DataCollection(nIds));
        }

        for (size_t shift = 0; shift < nShifts; shift++)
        {
            size_t partnerId = ((curId - beginId) ^ shift) + beginId;
            if (partnerId < beginId || partnerId >= endId)
            {
                continue;
            }

            NumericTablePtr partnerTable;

            if (partnerId == curId)
            {
                partnerTable = table;
            }
            else
            {
                if (curId < partnerId)
                {
                    sendTable(table, partnerId, tag);
                    recvTable(partnerTable, partnerId, tag);
                }
                else
                {
                    recvTable(partnerTable, partnerId, tag);
                    sendTable(table, partnerId, tag);
                }
            }

            if (partnerTable.get() && partnerTable->getNumberOfRows() > 0)
            {
                if (preserveOrder)
                {
                    (*destCollection)[partnerId - beginId] = partnerTable;
                }
                else
                {
                    destCollection->push_back(partnerTable);
                }
            }
        }
    }

    void sendTableAllToMaster(size_t beginId, size_t endId, size_t rankId, int tag, NumericTablePtr & table, DataCollectionPtr & destCollection)
    {
        if (rankId == beginId)
        {
            for (size_t partnerId = beginId; partnerId < endId; partnerId++)
            {
                NumericTablePtr partnerTable;
                if (partnerId == rankId)
                {
                    partnerTable = table;
                }
                else
                {
                    recvTable(partnerTable, partnerId, tag);
                }

                if (partnerTable.get() && partnerTable->getNumberOfRows() > 0)
                {
                    destCollection->push_back(partnerTable);
                }
            }
        }
        else
        {
            sendTable(table, beginId, tag);
        }
    }

    void sendCollectionMasterToAll(size_t beginId, size_t endId, size_t rankId, int tag, DataCollectionPtr & collection, NumericTablePtr & destTable)
    {
        if (rankId == beginId)
        {
            for (size_t partnerId = beginId; partnerId < endId; partnerId++)
            {
                NumericTablePtr table = NumericTable::cast((*collection)[partnerId - beginId]);
                if (partnerId == rankId)
                {
                    destTable = table;
                }
                else
                {
                    sendTable(table, partnerId, tag);
                }
            }
        }
        else
        {
            recvTable(destTable, beginId, tag);
        }
    }

    void sendTableMasterToAll(size_t beginId, size_t endId, size_t rankId, int tag, NumericTablePtr & table, NumericTablePtr & destTable)
    {
        if (rankId == beginId)
        {
            for (size_t partnerId = beginId; partnerId < endId; partnerId++)
            {
                if (partnerId == rankId)
                {
                    destTable = table;
                }
                else
                {
                    sendTable(table, partnerId, tag);
                }
            }
        }
        else
        {
            recvTable(destTable, beginId, tag);
        }
    }

    void sendTable(NumericTablePtr & table, int recpnt, int tag) { tcvr->send<NumericTablePtr>(table, recpnt, tag * 2); }

    void recvTable(NumericTablePtr & table, int sender, int tag) { table = tcvr->recv<NumericTablePtr>(sender, tag * 2); }
};

// oneDAL Distributed algos do not return a proper result (like batch), we need to create one
template <typename fptype, daal::algorithms::dbscan::Method method>
typename dbscan_manager<fptype, method>::iomb_type::result_type make_result(const daal::data_management::NumericTablePtr & assignments, const daal::data_management::NumericTablePtr & nClusters)
{
    typename dbscan_manager<fptype, method>::iomb_type::result_type res(new typename dbscan_manager<fptype, method>::iomb_type::result_type::ElementType);
    res->set(daal::algorithms::dbscan::assignments, daal::data_management::convertToHomogen<fptype>(*assignments.get()));
    res->set(daal::algorithms::dbscan::nClusters, daal::data_management::convertToHomogen<fptype>(*nClusters.get()));
    return res;
}

template <typename fptype, daal::algorithms::dbscan::Method method>
class dist_custom<dbscan_manager<fptype, method> >
{
public:
    typedef dbscan_manager<fptype, method> Algo;

    template <typename T1>
    static typename Algo::iomb_type::result_type _compute(Algo & algo, const T1 & input1)
    {
        dist<T1, fptype> d;
        d.epsilon         = algo._epsilon;
        d.minObservations = algo._minObservations;
        d.main(input1);
        return make_result<fptype, method>(d.assignments, d.totalNClusters);
    }

    template <typename... Ts>
    static typename Algo::iomb_type::result_type compute(Algo & algo, const Ts &... inputs)
    {
        return _compute(algo, get_table(inputs)...);
    }
};

} // namespace dist_custom

#endif // _DIST_DBSCAN_INCLUDED_

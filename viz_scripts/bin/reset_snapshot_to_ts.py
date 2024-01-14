## This is a standalone script that resets the database to a previously
## specified timestamp. It does so by deleting entries after the timestamp,
## and removing labels that were entered after the timestamp as well

import argparse
import arrow
import emission.core.get_database as edb
import emission.storage.decorations.user_queries as ecdu
import emission.core.wrapper.entry as ecwe
import emission.core.wrapper.pipelinestate as ecwp

import emission.storage.decorations.trip_queries as esdt
import emission.storage.timeseries.abstract_timeseries as esta

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="export_participants_trips")
    parser.add_argument("reset_fmt_time", help="reset timestamp in the format displayed on the metrics you are trying to match - e.g. 'YYYY-MM-DDThh:mm:ss.ms+00:00'")

    args = parser.parse_args()
    reset_ts = arrow.get(args.reset_fmt_time).timestamp()
    print(f"After parsing, the reset timestamp is {args.reset_fmt_time} -> {reset_ts}")

#     for user_id in ecdu.get_all_uuids():
#         mi_pipeline_state = edb.get_pipeline_state_db().find_one({"user_id": user_id,
#             "pipeline_stage": ecwp.PipelineStages.USER_INPUT_MATCH_INCOMING.value})
#         if mi_pipeline_state is None:
#             print(f"for {user_id}, USER_INPUT_MATCH_INCOMING was never run")
#         else:
#             print(f"for {user_id}, USER_INPUT_MATCH_INCOMING was last run at {arrow.get(mi_pipeline_state['last_run_ts'])}")
# 
#         if cc_pipeline_state is None:
#             print(f"for {user_id}, USER_INPUT_MATCH_INCOMING was never run")
#         cc_pipeline_state = edb.get_pipeline_state_db().find_one({"user_id": user_id,
#             "pipeline_stage": ecwp.PipelineStages.CREATE_CONFIRMED_OBJECTS.value})
#         print(f"for {user_id}, USER_INPUT_MATCH_INCOMING was last run at {arrow.get(cc_pipeline_state['last_run_ts'])}")
# 
    #first, count all entries written after a cutoff time
    print(f"Planning to delete {edb.get_timeseries_db().count_documents({ 'metadata.write_ts': { '$gt': reset_ts } })} records from the timeseries")
    print(f"Planning to delete {edb.get_analysis_timeseries_db().count_documents({ 'metadata.write_ts': { '$gt': reset_ts } })} records from the analysis timeseries")
    print(edb.get_analysis_timeseries_db().count_documents({ "metadata.write_ts": { "$gt": reset_ts } }))
    
    #then, find manual inputs added after the cutoff time
    print(f"number of manual records after cutoff {(edb.get_timeseries_db().count_documents({'metadata.write_ts': {'$gt': reset_ts}, 'metadata.key': {'$regex': '^manual/(mode_confirm|purpose_confirm|replaced_mode)$'}}))}")
    
    # ideally, this would use the aggreate timeseries, but retaining this code to make it easier for
    # @iantei to understand
    ts = esta.TimeSeries.get_aggregate_time_series()
    for t in list(edb.get_timeseries_db().find({"metadata.write_ts": {"$gt": reset_ts}, "metadata.key": {"$regex": '^manual/(mode_confirm|purpose_confirm|replaced_mode)$'}}).sort("metadata.write_ts", 1)):
        confirmed_trip = esdt.get_confirmed_obj_for_user_input_obj(ts, ecwe.Entry(t))
        if confirmed_trip is None:
            print("No matching confirmed trip for %s" % t["data"]["start_fmt_time"])
            continue
 
        if confirmed_trip["data"]["user_input"] == {}:
            print(f"For input {t['data']['start_fmt_time']} of type {t['metadata']['key']}, labeled at {t['metadata']['write_fmt_time']}, found confirmed trip starting at {confirmed_trip['data']['start_fmt_time']} with no user input")
        else:
            print(f"For input {t['data']['start_fmt_time']} of type {t['metadata']['key']}, labeled at {t['metadata']['write_fmt_time']}, found confirmed trip starting at {confirmed_trip['data']['start_fmt_time']} with user input {confirmed_trip['data']['user_input']}")
            input_type = t["metadata"]["key"].split("/")[1]
            print(f"Resetting input of type {input_type}")
            update_results = edb.get_analysis_timeseries_db().update_one({"_id": confirmed_trip["_id"],
                "metadata.key": "analysis/confirmed_trip"}, { "$unset": { 'data.user_input.%s' % input_type: {} } })
            print(f"Update results = {update_results.raw_result}")

    print(f"delete all entries after timestamp {reset_ts}")
    print(f"deleting all timeseries entries after {reset_ts}, {edb.get_timeseries_db().delete_many({ 'metadata.write_ts': { '$gt': reset_ts } }).raw_result}")
    print(f"deleting all analysis timeseries entries after {reset_ts}, {edb.get_analysis_timeseries_db().delete_many({ 'metadata.write_ts': { '$gt': reset_ts } }).raw_result}")


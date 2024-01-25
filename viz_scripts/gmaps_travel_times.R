library(ggmap)
library(gmapsdistance)
library(Hmisc)
library(tmaptools)

data = readRDS("apollo_data.rds")
set.api.key("")

# This part is for sampling the data, to reduce the cost (should be commented out in full run)
# set.seed(42)
# keep_IDs = data %>%
#   group_by(
#     ID
#   ) %>%
#   summarise(
#     n = n()
#   ) %>%
#   filter(
#     n > 100
#   )
# data = data %>%
#   filter(
#     ID %in% keep_IDs$ID
#   ) %>%
#   group_by(
#     ID
#   ) %>%
#   slice_sample(
#     n = 10
#   )

# Match study modes to Google API
data$galt = NA
data[data$alt %in% c("car","scar"),]$galt = "driving"
data[data$alt %in% c("walk"),]$galt = "walking"
data[data$alt %in% c("pmicro","ebike"),]$galt = "bicycling"
data[data$alt %in% c("transit"),]$galt = "transit"
unique(data$galt)
print(4*nrow(data)*.005)

backup_data = data
data = backup_data[5001:11903,]

# Get distance between origin and destination
# TRPMILES is calculated as distance on Google Maps by NHTS
modes = c("driving","bicycling","walking","transit")
result = list()
for (mode_sel in modes) {
  print(mode_sel)
  all_times = c()
  all_dists = c()
  all_statuses = c()
  for (i in 1:nrow(data)) {
    od = data[i,]
    if (is.na(od$olat)) {
      all_times = append(all_times, NA)
      all_dists = append(all_dists, NA)
      all_statuses = append(all_statuses, NA)
    } else {
      dist_results = tryCatch({
        gmapsdistance(origin=paste0(od$olat,"+",od$olon),
                      destination=paste0(od$dlat,"+",od$dlon),
                      mode=mode_sel,
                      combinations="pairwise")
      }, error = function(e){
        list(Time=NA,Distance=NA,Status="ERROR")
      })
      all_times = append(all_times, dist_results$Time)
      all_dists = append(all_dists, dist_results$Distance)
      all_statuses = append(all_statuses, dist_results$Status)
    }
  }
  result[[paste0("gtt_", mode_sel)]] = all_times
  result[[paste0("gd_", mode_sel)]] = all_dists
  result[[paste0("gs_", mode_sel)]] = all_statuses
}

# Units are meters for distance, seconds for time
apollo_data_gcoded = cbind(data, data.frame(result))
saveRDS(apollo_data_gcoded, file='./apollo_data_gcoded.rds')

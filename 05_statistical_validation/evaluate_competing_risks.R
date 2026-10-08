#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(cmprsk))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript evaluate_competing_risks.R INPUT.csv OUTPUT_DIR")
}

input_csv <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

prepare_competing_data <- function(data) {
  data$cr_status <- as.integer(data$event_type)
  data$cr_time_months <- as.numeric(data$analysis_time_months)
  if (anyNA(data$cr_status) || any(!is.finite(data$cr_time_months)) ||
      any(!data$cr_status %in% c(0L, 1L, 2L)) || any(data$cr_time_months <= 0)) {
    stop("Invalid adjudicated event records")
  }
  data$Risk_Group <- factor(data$Risk_Group, levels = c("Low", "High"))
  if (anyNA(data$Risk_Group)) stop("Risk_Group must be Low or High")
  data[is.finite(data$cr_time_months) & !is.na(data$Risk_Group), ]
}

summarize_subset <- function(data, subset_name) {
  subset <- prepare_competing_data(data[data$Subset == subset_name, ])
  if (nrow(subset) == 0 || length(unique(subset$Risk_Group)) < 2) {
    return(data.frame(Subset = subset_name, N = nrow(subset),
                      EHR_Events = sum(subset$cr_status == 1),
                      Competing_Deaths = sum(subset$cr_status == 2),
                      Censored = sum(subset$cr_status == 0), Gray_P = NA_real_))
  }
  fit <- cmprsk::cuminc(
    ftime = subset$cr_time_months,
    fstatus = subset$cr_status,
    group = subset$Risk_Group,
    cencode = 0
  )
  gray_p <- if (!is.null(fit$Tests) && "1" %in% rownames(fit$Tests)) {
    as.numeric(fit$Tests["1", "pv"])
  } else {
    NA_real_
  }
  data.frame(
    Subset = subset_name,
    N = nrow(subset),
    EHR_Events = sum(subset$cr_status == 1),
    Competing_Deaths = sum(subset$cr_status == 2),
    Censored = sum(subset$cr_status == 0),
    Gray_P = gray_p
  )
}

data <- read.csv(input_csv, check.names = FALSE)
required <- c(
  "Subset", "Risk_Group", "event_type", "analysis_time_months"
)
missing <- setdiff(required, names(data))
if (length(missing) > 0) {
  stop(paste("Missing columns:", paste(missing, collapse = ", ")))
}

subsets <- unique(as.character(data$Subset))
summary <- do.call(rbind, lapply(subsets, function(x) summarize_subset(data, x)))
write.csv(summary, file.path(output_dir, "competing_risk_summary.csv"), row.names = FALSE)

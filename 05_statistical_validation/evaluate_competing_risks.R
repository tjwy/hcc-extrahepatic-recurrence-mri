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
  data$Distant_Metastasis <- as.integer(data$Distant_Metastasis)
  data$OS_Status <- as.integer(data$OS_Status)
  data$Time_to_DM_Months <- as.numeric(data$Time_to_DM_Months)
  data$OS_Months <- as.numeric(data$OS_Months)
  data$cr_status <- ifelse(
    data$Distant_Metastasis == 1,
    1L,
    ifelse(data$OS_Status == 1, 2L, 0L)
  )
  data$cr_time_months <- ifelse(
    data$cr_status == 1L,
    data$Time_to_DM_Months,
    ifelse(data$cr_status == 2L, data$OS_Months, data$Time_to_DM_Months)
  )
  data$Risk_Group <- factor(data$Risk_Group, levels = c("Low/intermediate", "High"))
  data[is.finite(data$cr_time_months) & !is.na(data$Risk_Group), ]
}

summarize_subset <- function(data, subset_name) {
  subset <- prepare_competing_data(data[data$Subset == subset_name, ])
  if (nrow(subset) == 0 || length(unique(subset$Risk_Group)) < 2) {
    return(data.frame(Subset = subset_name, N = nrow(subset), Gray_P = NA_real_))
  }
  fit <- cmprsk::cuminc(
    ftime = subset$cr_time_months,
    fstatus = subset$cr_status,
    group = subset$Risk_Group,
    cencode = 0
  )
  gray_p <- if (!is.null(fit$Tests) && nrow(fit$Tests) > 0) {
    as.numeric(fit$Tests[1, "pv"])
  } else {
    NA_real_
  }
  data.frame(
    Subset = subset_name,
    N = nrow(subset),
    DM_Events = sum(subset$cr_status == 1),
    Competing_Deaths = sum(subset$cr_status == 2),
    Censored = sum(subset$cr_status == 0),
    Gray_P = gray_p
  )
}

data <- read.csv(input_csv, check.names = FALSE)
required <- c(
  "Subset", "Risk_Group", "Distant_Metastasis", "Time_to_DM_Months",
  "OS_Status", "OS_Months"
)
missing <- setdiff(required, names(data))
if (length(missing) > 0) {
  stop(paste("Missing columns:", paste(missing, collapse = ", ")))
}

subsets <- unique(as.character(data$Subset))
summary <- do.call(rbind, lapply(subsets, function(x) summarize_subset(data, x)))
write.csv(summary, file.path(output_dir, "competing_risk_summary.csv"), row.names = FALSE)

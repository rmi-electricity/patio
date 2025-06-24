# Patio Economic Model: V2

# Table of Contents:
## Step 1: Asset Cost Integration
### Step 1a: Clean and Modify Asset Dispatch Data
### Step 1b: Prep tidy dataframe
### Step 1c: Pull in utility-specific data
## Step 2: Loops to calculate costs by balancing authority
### Step 2a: Renewable cost calculations
### Step 2b: Fossil cost calculations
### Step 2c: Combine renewable and fossil cost data
## Step 3: Optimizing Cost-Effective Scenarios
### Step 3a: (Incomplete, bugs still need to be worked out)

# Set User (clunky way of navigating different working directories, not sure how to improve)
user <- Sys.info()["user"]
patio_results <- "202504270143"
# resource_results <- "BAs_202406121704_results"
resource_results <- "202504270143"
run_date <- "202406121704"
FRED_API_KEY <- "80d95a436d246849c5ce40d45361feb3"
# BLS_KEY <- "693c55f2f3b447fabf691ff5edf67ee6"
BLS_KEY <- "fbedb7ac1c34445a8a1ee7296b5cb60a"
IRA_on <- TRUE
baseline_year <- 2021
baseline_emissions_year <- 2021
EIR_NewERA_fin_build_year <-2031
final_build_year <- 2038
target_build_year <- EIR_NewERA_fin_build_year
final_loop_year <- EIR_NewERA_fin_build_year
irp_year <- 2024

# Load packages
library(RCurl)
library(lubridate)
library(arrow)
library(readxl)
library(dplyr)
library(lmtest)
library(ggplot2)
library(broom)
library(tidyr)
library(tidyxl)
library(janitor)
library(purrr)
library(beepr)
library(stringr)
library(RSQLite)
library(bit64)
library(fredr)
library(blsR)
library(reticulate)
library(jsonlite)

# Step 1: Renewable Cost Integration #####

# Step 1a: Clean and Modify Renewable Dispatch Data
## Pull in data
### Pull in resource model data

# python integration instructions here: https://rmi.github.io/etoolbox/etb_and_r.html#setup-etoolbox-with-uv

use_python("~/.local/share/uv/tools/rmi-etoolbox/bin/python")
cloud <- import("etoolbox.utils.cloud")
results <- cloud$read_patio_resource_results(resource_results)

 # if(user == "Cfong") {
 #     setwd(paste("~/RMI/Utility Transition Finance - Clean Repowering/Analysis/resource model results",run_date,resource_results,sep="/"))
 # } else if (user == "udayvaradarajan") {
 #   uvwd = "/Users/udayvaradarajan/My Drive/GitHub/patio-model"
 #   uvpatiodir = paste(uvwd, "patio_out/",patio_results,"/", sep="/")
 #   if (!grepl(uvpatiodir,getwd())) setwd(uvpatiodir)
 # } else {
 #   "set file path for user"
 # }

summary_parquet <- results$summary
allocated_parquet <- results$allocated %>%
  rename(NREL_Class = class_atb) %>%
  mutate(re_limits_dispatch = if_else(re_energy<=0,TRUE,re_limits_dispatch),
         plant_id_eia = as.integer64(plant_id_eia),
         NREL_Class = as.factor(if_else(!is.na(NREL_Class),paste("Class",NREL_Class, sep=""),NA))
         )
system_parquet <- results$system  %>%
  mutate(
    # re_limits_dispatch = unlist(re_limits_dispatch),
    re_limits_dispatch = if_else(re_energy<=0,TRUE,re_limits_dispatch),
    storage_li_pct = if_else(grepl("0.25",scenario),0.25,0)
  )

full_parquet <- results$full  %>%
  rename(NREL_Class = class_atb) %>%
  mutate(re_limits_dispatch = if_else(re_energy<=0,TRUE,re_limits_dispatch),
         plant_id_eia = as.integer64(plant_id_eia),
         NREL_Class = as.factor(if_else(!is.na(NREL_Class),paste("Class",NREL_Class, sep=""),NA))
         )

valid_ba_scens <- full_parquet %>%
  select(c(ba_code,scenario)) %>% distinct()

# Break the allocated parquet new asset deployments into phases and introduce "re_generator_id" to track each phase seperately

allocated_parquet <- allocated_parquet %>%
  mutate(capacity_mw = round(signif(capacity_mw, 6), 0)) %>%
  filter(capacity_mw>0) %>%
  group_by(ba_code,re_plant_id,re_generator_id,plant_id_eia,generator_id,year,technology_description,NREL_Class) %>%
  mutate(
    num_scens = n_distinct(scenario),
    num_phases = n_distinct(capacity_mw)
  ) %>% ungroup() %>%
  group_by(ba_code,re_plant_id,plant_id_eia,generator_id,year,technology_description,NREL_Class) %>%
  arrange(num_scens,capacity_mw,re_generator_id, .by_group = TRUE) %>%
  mutate(
    capacity_change = capacity_mw - if_else(num_phases>1,coalesce(lag(capacity_mw,1),0),0),
    rank = as.integer(if_else(num_phases>1,cumsum(capacity_change>=1),cumsum(if_else(is.na(lag(re_generator_id,1)),1,re_generator_id!=lag(re_generator_id,1)))))
  ) %>% ungroup() %>%
  group_by(ba_code,re_plant_id,plant_id_eia,generator_id,year,technology_description,NREL_Class,rank) %>%
  mutate(capacity_change = max(capacity_change)) %>% ungroup() %>%
  filter(capacity_change>0) %>%
  group_by(ba_code,re_plant_id,plant_id_eia,generator_id,year,technology_description,NREL_Class) %>%
  mutate(capacity_frac = capacity_change/max(capacity_mw)) %>% ungroup()

patio_cap_changes <- allocated_parquet %>%
  filter(num_phases>1) %>%
  select(c(ba_code,re_plant_id,plant_id_eia,generator_id,year,technology_description,NREL_Class,re_generator_id_new = rank,capacity_change)) %>%
  filter(capacity_change>=1) %>%
  distinct()

allocated_parquet <- allocated_parquet %>%
  select(-c(capacity_change,capacity_frac)) %>%
  left_join(patio_cap_changes, relationship = "many-to-many") %>%
  filter(is.na(re_generator_id_new) | ((!is.na(re_generator_id_new)) & rank>=re_generator_id_new)) %>%
  mutate(
    re_generator_id_old = re_generator_id,
    technology_id =  as.integer(unclass(as.factor(technology_description))-1),
    re_generator_id = as.integer(coalesce(re_generator_id_new,rank) + technology_id*100),
    capacity_frac = coalesce(capacity_change,capacity_mw)/capacity_mw,
    capacity_mw = coalesce(capacity_change,capacity_mw),
    across(c(redispatch_mwh:implied_need_mwh,redispatch_mmbtu:redispatch_cost_startup),
           ~ .x * capacity_frac),
    capacity_mw = round(signif(capacity_mw, 6), 0)
  ) %>% select(-c(rank,capacity_frac,re_generator_id_new)) %>%
  filter(capacity_mw>=1) %>%
  group_by(re_plant_id,re_generator_id,year) %>%
  mutate(test = n_distinct(capacity_mw)) %>% ungroup()

allocated_parquet <- allocated_parquet %>%
  select(-c(num_scens,num_phases,capacity_change,re_generator_id_old,technology_id,test)) %>%
  inner_join(valid_ba_scens, by = c("ba_code","scenario"))

full_parquet_map <- full_parquet %>%
  select(plant_id_eia,generator_id,utility_id_eia) %>% distinct()

### Pull in BBB Inputs
if(user == "Cfong") {
  setwd("~/Library/CloudStorage/OneDrive-RMI/Documents/git/patio-model/r_data")
} else if (user == "udayvaradarajan") {
  if (!grepl("r_data",getwd())) setwd(paste(uvwd, "r_data/", sep="/"))
} else {
  "set file path for user"
}

trans_costs <- as.numeric(read.csv("tx_dollars_per_mw_total.csv") %>% select(c(dollars_per_capacity_difference)))

### Read in named ranges from Excel model front-end and assign single cell ranges to R scalar variables, tables to R dataframes.
### These named ranges provide critical policy, storage, and tax assumptions for the analysis.

model_inputs<-xlsx_names("BBB Fossil Transition Analysis Inputs.xlsm") %>%
  subset(is_range == TRUE & hidden == FALSE, select = c(name,formula))

named_ranges<-model_inputs$name
for (range_name in named_ranges) {
  range_formula<-model_inputs %>%
    subset(name==range_name,select="formula") %>% as.character
  if (!grepl(":",range_formula, fixed = TRUE)) {
    range_value_test<-read_excel("BBB Fossil Transition Analysis Inputs.xlsm",range = range_formula,col_names=FALSE)
    range_value = ifelse(is.na(as.numeric(range_value_test)),as.character(range_value_test),as.numeric(range_value_test))
    assign(range_name,range_value)
  }
  else if (grepl("Toggles",range_formula, fixed = TRUE) |
           grepl("Tax Depreciation",range_formula, fixed = TRUE) |
           grepl("BLS Data Series",range_formula, fixed = TRUE)) {
    range_value_test<-read_excel("BBB Fossil Transition Analysis Inputs.xlsm",range = range_formula)
    range_value_test<-as.data.frame(unclass(range_value_test), stringsAsFactors = TRUE, optional = TRUE)
    assign(range_name,range_value_test)
  }
}

parameters <- read_excel("BBB Fossil Transition Analysis Inputs.xlsm", "Toggles", col_names=FALSE)

CPIU <- CPIU[c("Year","Inflation_Factor_2021")]

### Set parameters
build_years <- c(baseline_year:final_build_year)
irp_years <- c((irp_year+1):(irp_year+NPV_Duration))
all_years <- c((baseline_year):(irp_year+NPV_Duration))
all_tenors <- c(1:50)
debt_years <- c(1:Debt_Tenor)
nrel_atb_years <- c((baseline_year):final_build_year)
technology_columns <- c("LandbasedWind",
                        "UtilityPV",
                        "Utility-Scale Battery Storage",
                        "Pumped Storage Hydropower",
                        "OffShoreWind",
                        "Geothermal",
                        "Nuclear",
                        "Coal_FE",
                        "NaturalGas_FE",
                        "Coal_Retrofits",
                        "NaturalGas_Retrofits"

)

cur_year <- year(today())
cur_month <- month(today())
cur_day <- day(today())

data_start_date <- as.Date(ifelse(cur_day>10,make_date(cur_year,cur_month,1),
                          ifelse(cur_month==1,make_date(cur_year-1,12,1),
                                 make_date(cur_year,cur_month-1,1))))

fredr_set_key(FRED_API_KEY)
T5YIE <- fredr(series="T5YIE", observation_start = data_start_date)
T5YIFR <- fredr(series="T5YIFR", observation_start = data_start_date)
BAMLC0A1CAAA <- fredr(series="BAMLC0A1CAAA", observation_start = data_start_date)
BAMLC0A2CAA <- fredr(series="BAMLC0A2CAA", observation_start = data_start_date)
BAMLC0A3CA <- fredr(series="BAMLC0A3CA", observation_start = data_start_date)
BAMLC0A4CBBB <- fredr(series="BAMLC0A4CBBB", observation_start = data_start_date)
BAMLH0A1HYBB <- fredr(series="BAMLH0A1HYBB", observation_start = data_start_date)
BAMLH0A2HYB <- fredr(series="BAMLH0A2HYB", observation_start = data_start_date)
DGS1MO <- fredr(series="DGS1MO", observation_start = data_start_date)
#DGS2MO <- fredr(series="DGS2MO", observation_start = data_start_date)
DGS3MO <- fredr(series="DGS3MO", observation_start = data_start_date)
#DGS4MO <- fredr(series="DGS4MO", observation_start = data_start_date)
DGS6MO <- fredr(series="DGS6MO", observation_start = data_start_date)
DGS1 <- fredr(series="DGS1", observation_start = data_start_date)
DGS2 <- fredr(series="DGS2", observation_start = data_start_date)
DGS3 <- fredr(series="DGS3", observation_start = data_start_date)
DGS5 <- fredr(series="DGS5", observation_start = data_start_date)
DGS7 <- fredr(series="DGS7", observation_start = data_start_date)
DGS10 <- fredr(series="DGS10", observation_start = data_start_date)
DGS20 <- fredr(series="DGS20", observation_start = data_start_date)
DGS30 <- fredr(series="DGS30", observation_start = data_start_date)

bls_set_key(BLS_KEY)
CPIU_table <- get_series_table("CUUR0000SA0",start_year = 1913,end_year = cur_year)

forward_rates_name <- paste("forward_rates-",cur_year,
                            ifelse(cur_month>=10,cur_month,0),
                            ifelse(cur_month>=10,"",cur_month),
                            ifelse(cur_day>=10,cur_day,0),
                            ifelse(cur_day>=10,"",cur_day),
                            ".parquet",
                            sep="")

if(file.exists(forward_rates_name)) {
  Forward_Interest_Rates <- read_parquet(forward_rates_name)
} else {
  prev_month <- case_when(
    cur_month==1 ~ 12,
    TRUE ~ cur_month-1)

  prev_year <- case_when(
    cur_month==1 ~ cur_year-1,
    TRUE ~ cur_year)

  treasury_link <- paste("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/",
                         cur_year,ifelse(cur_month>=10,cur_month,0),ifelse(cur_month>=10,"",cur_month),
                         "?type=daily_treasury_yield_curve&field_tdr_date_value_month=",
                         cur_year,ifelse(cur_month>=10,cur_month,0),ifelse(cur_month>=10,"",cur_month),
                         "&page&_format=csv",
                         sep = "")

  treasury_link_prev <- paste("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/",
                              prev_year,ifelse(prev_month>=10,prev_month,0),ifelse(prev_month>=10,"",prev_month),
                              "?type=daily_treasury_yield_curve&field_tdr_date_value_month=",
                              prev_year,ifelse(prev_month>=10,prev_month,0),ifelse(prev_month>=10,"",prev_month),
                              "&page&_format=csv",
                              sep = "")


  if (cur_day>3) {

    treasury_yields_raw <- as.data.frame(rbind(read.csv(treasury_link),read.csv(treasury_link_prev)))

  } else{

    treasury_yields_raw <- read.csv(treasury_link_prev)

  }

  treasury_yields <- treasury_yields_raw %>%
    pivot_longer(X1.Mo:X30.Yr,names_to = "tenor", values_to = "yield") %>%
    mutate(
      tenor = if_else(grepl(".Mo",tenor),
                      as.numeric(sub("X","",sub(".Mo","",tenor)))/12,
                      as.numeric(sub("X","",sub(".Yr","",tenor)))),
      tenorsq = tenor^2,
      yield = yield/100
    )

  treasury_model <- lm(yield ~ tenor + tenorsq,data = treasury_yields)
  treasury_variables <- tidy(treasury_model) %>%
    select(c(term, estimate))

  intercept <- as.numeric(treasury_variables %>% filter(term=="(Intercept)") %>% select(estimate))
  tenor_coeff <- as.numeric(treasury_variables %>% filter(term=="tenor") %>% select(estimate))
  tenorsq_coeff <- as.numeric(treasury_variables %>% filter(term=="tenorsq") %>% select(estimate))

  Forward_Interest_Rates <- cross_join(as.data.frame(all_years),as.data.frame(all_tenors)) %>%
    rename("Years" = "all_years","Tenor" = "all_tenors") %>%
    mutate(
      Forward_Treasury_Rates = intercept + tenor_coeff * Tenor + tenorsq_coeff * Tenor^2,
      Spot_Rates_Issuance_Year = intercept + tenor_coeff * (Years-cur_year) + tenorsq_coeff * (Years-cur_year)^2,
      Weighted_Average_Tenor = (Tenor+1)-(1+Forward_Treasury_Rates)/Forward_Treasury_Rates +
        Tenor/(((1+Forward_Treasury_Rates)^Tenor)-1),
      Weighted_Average_Tenor_DOE = Weighted_Average_Tenor,
      Weighted_Average_Tenor_BBB = Weighted_Average_Tenor,
      Weighted_Average_Tenor_AAA = Weighted_Average_Tenor,
      Weighted_Average_Tenor_A = Weighted_Average_Tenor,
      Weighted_Average_Tenor_AA = Weighted_Average_Tenor,
      Weighted_Average_Tenor_B = Weighted_Average_Tenor,
      Weighted_Average_Tenor_BB = Weighted_Average_Tenor
    )

  for(i in 1:10){
    Forward_Interest_Rates <- Forward_Interest_Rates %>%
      mutate(
        Spot_Rates_at_Maturity = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor)^2,
        Forward_Treasury_Rates = (((1+Spot_Rates_at_Maturity)^(Years-cur_year+Weighted_Average_Tenor)) /
                                    ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor) - 1,
        Weighted_Average_Tenor = (Tenor+1)-(1+Forward_Treasury_Rates)/Forward_Treasury_Rates +
          Tenor/(((1+Forward_Treasury_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_DOE = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_DOE) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_DOE)^2,
        DOE_Loan_Rates = 0.00375 + (((1+Spot_Rates_at_Maturity_DOE)^(Years-cur_year+Weighted_Average_Tenor_DOE)) /
                                      ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_DOE) - 1,
        Weighted_Average_Tenor_DOE = (Tenor+1)-(1+DOE_Loan_Rates)/DOE_Loan_Rates +
          Tenor/(((1+DOE_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_BBB = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_BBB) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_BBB)^2,
        BBB_Loan_Rates = BBB_Spread + (((1+Spot_Rates_at_Maturity_BBB)^(Years-cur_year+Weighted_Average_Tenor_BBB)) /
                                      ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_BBB) - 1,
        Weighted_Average_Tenor_BBB = (Tenor+1)-(1+BBB_Loan_Rates)/BBB_Loan_Rates +
          Tenor/(((1+BBB_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_AAA = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_AAA) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_AAA)^2,
        AAA_Loan_Rates = AAA_Spread + (((1+Spot_Rates_at_Maturity_AAA)^(Years-cur_year+Weighted_Average_Tenor_AAA)) /
                                         ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_AAA) - 1,
        Weighted_Average_Tenor_AAA = (Tenor+1)-(1+AAA_Loan_Rates)/AAA_Loan_Rates +
          Tenor/(((1+AAA_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_AA = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_AA) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_AA)^2,
        AA_Loan_Rates = AA_Spread + (((1+Spot_Rates_at_Maturity_AA)^(Years-cur_year+Weighted_Average_Tenor_AA)) /
                                       ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_AA) - 1,
        Weighted_Average_Tenor_AA = (Tenor+1)-(1+AA_Loan_Rates)/AA_Loan_Rates +
          Tenor/(((1+AA_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_A = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_A) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_A)^2,
        A_Loan_Rates = A_Spread + (((1+Spot_Rates_at_Maturity_A)^(Years-cur_year+Weighted_Average_Tenor_A)) /
                                     ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_A) - 1,
        Weighted_Average_Tenor_A = (Tenor+1)-(1+A_Loan_Rates)/A_Loan_Rates +
          Tenor/(((1+A_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_BB = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_BB) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_BB)^2,
        BB_Loan_Rates = BB_Spread + (((1+Spot_Rates_at_Maturity_BB)^(Years-cur_year+Weighted_Average_Tenor_BB)) /
                                         ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_BB) - 1,
        Weighted_Average_Tenor_BB = (Tenor+1)-(1+BB_Loan_Rates)/BB_Loan_Rates +
          Tenor/(((1+BB_Loan_Rates)^Tenor)-1),
        Spot_Rates_at_Maturity_B = intercept + tenor_coeff * (Years-cur_year+Weighted_Average_Tenor_B) +
          tenorsq_coeff * (Years-cur_year+Weighted_Average_Tenor_B)^2,
        B_Loan_Rates = B_Spread + (((1+Spot_Rates_at_Maturity_B)^(Years-cur_year+Weighted_Average_Tenor_B)) /
                                         ((1+Spot_Rates_Issuance_Year)^(Years-cur_year)))^(1/Weighted_Average_Tenor_B) - 1,
        Weighted_Average_Tenor_B = (Tenor+1)-(1+B_Loan_Rates)/B_Loan_Rates +
          Tenor/(((1+B_Loan_Rates)^Tenor)-1)
      )
  }

  write_parquet(Forward_Interest_Rates,forward_rates_name)
}

irp_treas_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(Forward_Treasury_Rates)))
irp_BBB_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(BBB_Loan_Rates)))
irp_A_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(A_Loan_Rates)))
irp_AA_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(AA_Loan_Rates)))
irp_B_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(B_Loan_Rates)))
irp_BB_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(BB_Loan_Rates)))
irp_AAA_rate <- as.numeric(Forward_Interest_Rates %>% filter(Tenor==Debt_Tenor,Years==irp_year) %>% select(c(AAA_Loan_Rates)))

technology_columns_fossil <- c("Coal_FE",
                               "NaturalGas_FE",
                               "Coal_Retrofits",
                               "NaturalGas_Retrofits"
)

nrel_scenarios <-c(NREL_Scenario_tech, NREL_Scenario_financial)

## Read in NREL data
file = "ATBe.csv"
if (file.exists(file)) print(paste(file,' already downloaded')) else download.file("https://oedi-data-lake.s3.amazonaws.com/ATB/electricity/csv/2023/ATBe.csv",file)
nrel_atb <- read.csv(file,stringsAsFactors = TRUE)

### NREL parameters
core_metric_parameter_tech_columns <-
  c("CAPEX","CF","Fixed O&M","Variable O&M","Fuel", "OCC","Additional OCC",
    #"Heat Rate","Net Output Penalty",
    "Heat Rate Penalty")

core_metric_parameter_financial_columns <- c(
  #"Inflation Rate",
  "Calculated Rate of Return on Equity Real",
  "Calculated Interest Rate Real",
  "Debt Fraction",
  "Interest Rate Nominal",
  "Rate of Return on Equity Nominal",
  "WACC Nominal",
  "WACC Real")

nrel_atb_tech_intermediate <- nrel_atb %>%
  filter(core_metric_case == "Market",
         crpyears == 30) %>%
  filter(core_metric_parameter %in% core_metric_parameter_tech_columns) %>%
  filter(technology %in% technology_columns) %>%
  filter(core_metric_variable %in% nrel_atb_years) %>%
  filter(scenario %in% nrel_scenarios) %>%
  mutate(techdetail = if_else(techdetail=="",techdetail2,techdetail)) %>%
  select(technology, techdetail, core_metric_variable, core_metric_parameter, value)

### Pivot NREL table wide
nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
  pivot_wider(names_from = core_metric_parameter,
              values_from = value)

nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
  left_join(Tax_Equity_Params %>%
              select(c(
                Technology,
                Construction_Finance_Adder,
                Construction_Finance_Years
              )),
            by = c("technology" = "Technology")) %>%
  mutate(
    CAPEX = coalesce(CAPEX,
                     OCC + (OCC * Construction_Finance_Adder * (Construction_Finance_Years+1))/2,
                     `Additional OCC` + (`Additional OCC` * Construction_Finance_Adder * (Construction_Finance_Years+1))/2)
  ) %>% select(-c(OCC,`Additional OCC`))

### Only nuclear has VOM, 0 out others and subtract costs of new non-CCS fossil from the CCS costs
# as a rough estimate of CCS retrofit incremental capital and operating costs

nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
    mutate(
      `Variable O&M` = case_when(is.na(`Variable O&M`) ~ 0,
                                 TRUE ~ `Variable O&M`))

fossil_tech_combs <- nrel_atb_tech_intermediate %>%
  filter(technology=="Coal_Retrofits" | technology=="NaturalGas_Retrofits") %>%
  select(c(technology, techdetail)) %>% distinct()

nrel_atb_tech_intermediate_temp <- nrel_atb_tech_intermediate %>%
  filter(
    (technology == "Coal_FE" & techdetail =="New") | (technology == "NaturalGas_FE" & (techdetail %in% c("F-Frame CC","H-Frame CC")))) %>%
  mutate(
    `Variable O&M` = -`Variable O&M`,
    CAPEX = 0,
    `Fixed O&M` = -`Fixed O&M`,
    technology = if_else(technology == "Coal_FE", "Coal_Retrofits", "NaturalGas_Retrofits")
  ) %>%
  left_join(fossil_tech_combs %>% select(c(technology,techdetail2 = techdetail)), by = c("technology"), relationship = "many-to-many") %>%
  filter((grepl("F-F",techdetail2) & grepl("F-F",techdetail)) | (grepl("H-F",techdetail2) & grepl("H-F",techdetail)) | technology == "Coal_Retrofits") %>%
  select(-c(techdetail)) %>% rename("techdetail" = "techdetail2")

nrel_atb_tech_intermediate <- rbind(nrel_atb_tech_intermediate_temp,nrel_atb_tech_intermediate)

nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
  group_by(technology,techdetail,core_metric_variable) %>%
  summarize(
    `Variable O&M` = sum(`Variable O&M`),
    `Fixed O&M` = sum(`Fixed O&M`),
    CAPEX = sum(CAPEX),
    Fuel = sum(Fuel),
    CF = mean(CF),
    `Heat Rate Penalty` = sum(`Heat Rate Penalty`, na.rm = TRUE)
  ) %>% ungroup() %>% distinct()

rm(nrel_atb_tech_intermediate_temp)

## Calculate CF improvement, relative to NREL baseline year (2021 for NREL ATB 2023)

baseline_CF <- nrel_atb_tech_intermediate %>%
  filter(core_metric_variable == baseline_year) %>%
  select(-c(CAPEX, core_metric_variable, `Fixed O&M`, `Variable O&M`, Fuel,
            `Heat Rate Penalty`)) %>%
  rename(base_CF = CF) %>% filter(!is.na(base_CF))

nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
  left_join(baseline_CF,
            by = c("technology", "techdetail"))

nrel_atb_tech_intermediate <- nrel_atb_tech_intermediate %>%
  mutate(CF_improvement = if_else(is.na(base_CF) | base_CF==0,NA,CF / base_CF))

## Pull in PUDL 861 data for delivery utility cost impact calculations

# if(user == "Cfong") {
#   setwd("~/Library/CloudStorage/OneDrive-RMI/Documents/git/patio-model/pudl_outputs")
# } else if (user == "udayvaradarajan") {
#   if (!grepl("pudl_outputs",getwd())) setwd(paste(uvwd, "pudl_outputs/", sep="/"))
# } else {
#   "set file path for user"
# }
#
# file = "pudl.sqlite"
# if (file.exists(file)) print(paste(file,' already downloaded')) else print("download pudl.sqlite and place in pudl_outputs folder: https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/pudl.sqlite.zip")
#
# pudl <- dbConnect(SQLite(), file)
# pudl_dataframes <- dbListTables(pudl)

pudl_release <- results$pudl_release
pudl <- import("etoolbox.utils.pudl")
pudl$pudl_list(pudl_release)

utilities_eia860 <- pudl$pd_read_pudl("core_eia860__scd_utilities", release=pudl_release)

# utilities_eia860 <- dbReadTable(pudl, "core_eia860__scd_utilities") %>%
#   arrange(utility_id_eia) %>%
#   group_by(utility_id_eia) %>%
#   mutate(
#     has_entity_info = sum(!is.na(entity_type)),
#     is_newest = (year(report_date) == max(year(report_date) * ((has_entity_info>0 & !is.na(entity_type)) | has_entity_info==0),na.rm = TRUE)),
#     entity_type = as.factor(entity_type)
#   ) %>% filter((has_entity_info==0 & is_newest) | (is_newest & has_entity_info>0 & !is.na(entity_type)) ) %>% select(-c(has_entity_info)) %>% ungroup()

Utilities <- utilities_eia860 %>%
  select(c(
    utility_id_eia,
    entity_type_ER = entity_type
  )) %>% distinct()

# sales_ult_cust_861 <- pudl$pd_read_pudl("core_eia861__yearly_sales", release=pudl_release)
#
# # sales_ult_cust_861 <- dbReadTable(pudl, "core_eia861__yearly_sales") %>%
# #   filter(year(report_date) == 2022) %>%
# #   mutate_all(~replace_na(., 0))
#
# # test <- sales_ult_cust_861 %>%
# #   filter(service_type == "delivery")
# #
# # test <- sales_ult_cust_861 %>%
# #   filter(utility_id_eia == 733)
# #
# # test2 <- dbReadTable(pudl, "core_eia861__yearly_service_territory")
#
# sales_ult_cust_861_total <- sales_ult_cust_861 %>%
#   group_by(utility_id_eia, utility_name_eia, service_type, state, entity_type, balancing_authority_code_eia) %>%
#   summarize(total_customers = sum(customers),
#             total_sales_mwh = sum(sales_mwh),
#             total_sales_revenue = sum(sales_revenue))


# operational_data_861 <- dbReadTable(pudl, "core_eia861__yearly_operational_data_misc") %>%
#   filter(year(report_date) == 2022) %>%
#   mutate_all(~replace_na(., 0))
#
# ## calculate frac retail sales and frac sales for resale
#
# operational_data_861 <- operational_data_861 %>%
#   mutate(frac_retail_sales = retail_sales_mwh / (retail_sales_mwh + sales_for_resale_mwh),
#          frac_sales_for_resale = 1 - frac_retail_sales)
#

# dbDisconnect(pudl)

# Change directory to patio econ results to store all subsequent results
# make new folder in econ model results in sharepoint

if(user == "Cfong") {
  setwd(paste("~/RMI/Utility Transition Finance - Clean Repowering/Analysis/Economic Model Results",patio_results,sep="/"))
} else if (user == "udayvaradarajan") {
  uvwd = "/Users/udayvaradarajan/My Drive/GitHub/patio-model"
  # uvresultsdir = paste(uvwd,"/","econ_results/",patio_results,"/",sep="")
  uvresultsdir = paste("/Users/udayvaradarajan/Library/CloudStorage/OneDrive-RMI/Clean Repowering/Analysis/Economic Model Results/",patio_results,"/",sep = "")
  if(!dir.exists(uvresultsdir)) {dir.create(uvresultsdir)}
  setwd(uvresultsdir)
} else {
  "set file path for user"
}

# Pull in unit ownership and financial details and compile MUL of unique owned assets

master_unit_list <- read_parquet("unit_financial_inputs.parquet") %>% mutate( plant_id_eia = as.integer64(plant_id_eia))
asset_owners <- read_parquet("asset_owners.parquet")


# summary_parquet <- summary_parquet %>%
#   group_by(ba_code) %>%
#   mutate(
#     keep = (min(deficit_max_pct_net_load, na.rm = TRUE)<=0.1) & (min(deficit_gt_2pct_count, na.rm = TRUE)<=30)
#   )

# system_parquet <- system_parquet %>%
#   left_join(summary_parquet %>% select(c(
#     ba_code,
#     scenario,
#     keep
#   )), by = c("ba_code","scenario"))

### Trim the full parquet and split it into proposed and the rest

full_parquet <- full_parquet %>%
  filter(category != "patio_clean") %>%
  select(c(
    ba_code,
    scenario,
    plant_id_eia,
    generator_id,
    re_limits_dispatch,
    year,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    #storage_fe_pct,
    #storage_h2_pct,
    ccs_scen,
    #excl_or_moth,
    capacity_mw,
    redispatch_mwh,
    redispatch_curt_adj_mwh,
    redispatch_mmbtu,
    redispatch_co2_tonne,
    redispatch_cost_fuel,
    redispatch_cost_vom,
    redispatch_cost_startup,
    redispatch_cost_fom,
    implied_need_mw,
    #implied_need_mwh,
    category,
    utility_id_eia,
    technology_description,
    operational_year,
    retirement_year,
    energy_community,
    ccs_eligible,
    #historical_year,
    capex_per_kw,
    real_maint_capex_per_kw,
    prime_mover,
    NREL_Class,
    fuel_group
  )) %>%
  filter(capacity_mw > 0) %>%
  mutate(
    #operating_date = as_datetime(operating_date, tz = "GMT"),
    scenario = as.factor(scenario),
    ba_code = as.factor(ba_code),
    category = as.factor(category),
    technology_description = as.factor(technology_description),
    Technology_FERC = case_when(
      grepl("Batteries",technology_description) | grepl("Storage",technology_description) ~ "renewables",
      grepl("Nuclear",technology_description) ~ "nuclear",
      grepl("Hydro",technology_description) ~ "hydro",
      (grepl("Gas",technology_description) | grepl("Coal",technology_description) | grepl("Petroleum",technology_description)) & is.na(fuel_group) & prime_mover=="ST" ~ "steam",
      (grepl("Gas",technology_description) | grepl("Coal",technology_description) | grepl("Petroleum",technology_description)) & is.na(fuel_group) & prime_mover!="ST" ~ "other_fossil",
      (!is.na(fuel_group)) & (fuel_group %in% c("natural_gas","coal","petroleum","petroleum_coke","other_gas")) & prime_mover=="ST" ~ "steam",
      (!is.na(fuel_group)) & (fuel_group %in% c("natural_gas","coal","petroleum","petroleum_coke","other_gas")) & prime_mover!="ST" ~ "other_fossil",
      (technology_description %in% c("curtailment","deficit")) ~ NA,
      TRUE ~ "renewables"
    )
  )

full_proposed_parquet <- full_parquet %>%
  filter(category == "proposed_clean" | category == "proposed_fossil") %>%
  mutate(re_plant_id = plant_id_eia,
         re_generator_id = generator_id,
         distance = 0
  ) %>%
  select(-c(
    #excl_or_moth,
    redispatch_cost_vom,
    #redispatch_cost_startup,
    redispatch_cost_fom,
    #utility_id_eia,
    capex_per_kw,
    real_maint_capex_per_kw
  )) %>% group_by(year,plant_id_eia,generator_id) %>%
  mutate(
    redispatch_co2_tonne = coalesce(redispatch_co2_tonne,0),
    co2_reduced = if_else(redispatch_co2_tonne>0,
                                sum(if_else(scenario=="counterfactual",redispatch_co2_tonne,0), na.rm = TRUE) - redispatch_co2_tonne,0)
  ) %>% ungroup()


transmission_upgrades_patio_parquet <- full_parquet %>%
  filter(category=="existing_fossil") %>%
  mutate(re_plant_id = plant_id_eia,
         re_generator_id = generator_id,
         distance = 0,
         capacity_mw = implied_need_mw,
         redispatch_mwh = 0,
         redispatch_mmbtu = 0,
         redispatch_co2_tonne = 0,
         redispatch_cost_fuel = 0,
         redispatch_cost_startup = 0,
         co2_reduced = 0,
         category = "patio_transmission_upgrades",
         technology_description = "Transmission",
         Technology_FERC = "transmission",
         operational_year = NA,
         retirement_year = NA,
         energy_community = NA
  ) %>%
  select(-c(
    #excl_or_moth,
    redispatch_cost_vom,
    #redispatch_cost_startup,
    redispatch_cost_fom,
    #utility_id_eia,
    fuel_group,
    prime_mover,
    capex_per_kw,
    real_maint_capex_per_kw
  )) %>%
  group_by(
    scenario,
    plant_id_eia,
    generator_id
  ) %>% mutate(capacity_mw = max(capacity_mw, na.rm = TRUE)) %>% ungroup() %>%
  filter(capacity_mw>0)

### Keep only the fields needed for allocation

allocated_parquet <- allocated_parquet %>%
  select(c(
    ba_code,
    scenario,
    plant_id_eia,
    generator_id,
    re_plant_id,
    re_generator_id,
    #re_type,
    re_limits_dispatch,
    energy_community,
    year,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    #storage_fe_pct,
    #storage_h2_pct,
    ccs_scen,
    capacity_mw,
    redispatch_mwh,
    redispatch_curt_adj_mwh,
    redispatch_mmbtu,
    redispatch_co2_tonne = redispatch_co2,
    redispatch_cost_fuel,
    #redispatch_cost_vom,
    redispatch_cost_startup,
    #redispatch_cost_fom,
    implied_need_mw,
    #utility_id_eia,
    #implied_need_mwh,
    category,
    technology_description,
    operational_year,
    retirement_year,
    ccs_eligible,
    NREL_Class,
    #historical_year,
    distance
  )) %>%
  filter(capacity_mw > 0) %>%
  mutate(
    #operating_date = as_datetime(operating_date, tz = "GMT"),
    scenario = as.factor(scenario),
    ba_code = as.factor(ba_code),
    category = as.factor(category),
    co2_reduced = 0,
    technology_description = as.factor(technology_description),
    Technology_FERC = case_when(
      grepl("Nuclear",technology_description) ~ "nuclear",
      grepl("Batteries",technology_description) | grepl("Storage",technology_description) ~ "renewables",
      grepl("Hydro",technology_description) ~ "hydro",
      (grepl("Gas",technology_description) | grepl("Coal",technology_description)) & grepl("Steam",technology_description) ~ "steam",
      (grepl("Gas",technology_description) | grepl("Coal",technology_description) | grepl("Solid",technology_description) |
         grepl("Petroleum",technology_description) | grepl("Gas",technology_description) | grepl("All",technology_description)) ~ "other_fossil",
      (technology_description %in% c("curtailment","deficit")) ~ NA,
      TRUE ~ "renewables"
    )
  ) %>% left_join(full_parquet_map, by = c("plant_id_eia","generator_id"))

## Bind all proposed and patio new clean assets into the allocated_parquet and pull in NREL Class data

allocated_parquet <- rbind(allocated_parquet,
                           full_proposed_parquet %>% select(-c(fuel_group,prime_mover)),
                           transmission_upgrades_patio_parquet) %>%
  mutate(
    ba_code = as.factor(ba_code),
    scenario = as.factor(scenario),
    category = as.factor(category),
    technology_description = as.factor(technology_description),
    operational_year = as.numeric(operational_year),
    retirement_year = as.numeric(retirement_year)
  )

allocated_parquet <- allocated_parquet %>%
  mutate(
    technology_NREL = as.factor(case_when(
      technology_description == "Solar Photovoltaic" ~ "UtilityPV",
      technology_description == "Onshore Wind Turbine"  ~ "LandbasedWind",
      technology_description == "Offshore Wind Turbine"  ~ "OffShoreWind",
      technology_description == "Nuclear" ~ "Nuclear",
      technology_description == "Batteries" ~ "Utility-Scale Battery Storage",
      technology_description == "Hydroelectric Pumped Storage" ~ "Pumped Storage Hydropower",
      technology_description == "Geothermal" ~ "Geothermal",
      (technology_description == "Conventional Steam Coal") & ccs_eligible & ccs_scen!=0 ~ "Coal_Retrofits",
      (technology_description == "Natural Gas Fired Combined Cycle") & ccs_eligible & ccs_scen!=0 ~ "NaturalGas_Retrofits",
      grepl("Coal",technology_description) ~ "Coal_FE",
      grepl("Gas",technology_description) ~ "NaturalGas_FE",
      grepl("Petroleum",technology_description) ~ "NaturalGas_FE",
      technology_description == "Transmission" ~ "Transmission",
      TRUE ~ NA
    ))
  ) %>%
  left_join(Tax_Equity_Params %>%
              select(c(
                Technology,
                CCS_CO2_Frac,
              )),
            by = c("technology_NREL" = "Technology"))

allocated_parquet <- allocated_parquet %>%
  mutate(
    NREL_Class = coalesce(NREL_Class,as.factor(
      case_when((technology_description == "Conventional Steam Coal") &
                  ccs_eligible & ccs_scen!=0 ~ paste(CCS_CO2_Frac * 100,"%-CCS",sep=""),
                (technology_description == "Natural Gas Fired Combined Cycle") &
                  ccs_eligible & ccs_scen!=0 & operational_year<=2015 ~ paste("F-Frame CC ",CCS_CO2_Frac * 100,"%-CCS",sep=""),
                (technology_description == "Natural Gas Fired Combined Cycle") &
                  ccs_eligible & ccs_scen!=0 & operational_year>=2015 ~ paste("H-Frame CC ",CCS_CO2_Frac * 100,"%-CCS",sep=""),
                (technology_description == "Conventional Steam Coal") & category=="proposed_fossil" ~ "New",
                (technology_description == "Natural Gas Fired Combined Cycle") & category=="proposed_fossil" ~ "H-Frame CC",
                (technology_description == "Natural Gas Fired Combustion Turbine") & category=="proposed_fossil" ~ "F-Frame CT",
                technology_description == "Geothermal" ~ "HydroFlash",
                technology_description == "Pumped Storage Hydropower" ~ "NatlClass3",
                technology_description == "Nuclear" ~ "NuclearSMR",
                technology_description == "Batteries" ~ "4Hr Battery Storage",
                TRUE ~ NA))),
    cf_actual = redispatch_mwh / (capacity_mw * (8760+if_else(year %% 4 == 0,24,0)))
  ) %>% select(-c(technology_description)) %>% rename("technology_description" = "technology_NREL")

allocated_parquet <- allocated_parquet %>%
  mutate(
    operational_year = as.numeric(operational_year),
    retirement_year = as.numeric(retirement_year)
  )

renewable_nrel_class <- allocated_parquet %>%
  filter(redispatch_mwh>0,
         category=="proposed_clean",
         capacity_mw>0,
         technology_description %in% c("UtilityPV","LandbasedWind","OffShoreWind")) %>%
  group_by(ba_code,plant_id_eia,generator_id,re_plant_id,re_generator_id,category,scenario,technology_description) %>%
  summarize(
    cf_median = median(if_else(redispatch_mwh>0 & capacity_mw >0,cf_actual,NA), na.rm = TRUE)
  ) %>%
  left_join(baseline_CF, by = c("technology_description" = "technology"), relationship = "many-to-many") %>%
  mutate(
    ren_NREL_Class = techdetail[which.min(abs(cf_median-base_CF))]
  ) %>% ungroup() %>% filter(ren_NREL_Class == techdetail) %>% select(-c(techdetail))

allocated_parquet <- allocated_parquet %>%
  left_join(renewable_nrel_class %>% select(c(ba_code,plant_id_eia,generator_id,re_plant_id,re_generator_id,category,scenario,technology_description,ren_NREL_Class)),
            by = c("ba_code","re_plant_id","plant_id_eia","re_generator_id","generator_id","category","scenario","technology_description")) %>%
  mutate(NREL_Class = as.factor(coalesce(NREL_Class,ren_NREL_Class))) %>% select(-c(ren_NREL_Class))

allocated_parquet <- allocated_parquet %>%
  # left_join(system_parquet %>% select(c(
  #   ba_code,
  #   scenario,
  #   re_curtailment = re_curtailment_pct,
  #   non_re_curtailment = non_re_curtailment_pct_of_non_re_gen,
  #   year,
  #   re_limits_dispatch)), by = c("ba_code","scenario","year","re_limits_dispatch")) %>%
  mutate(
    technology_description = as.factor(technology_description),
    MWh_no_curt = redispatch_mwh,
    redispatch_mwh = redispatch_curt_adj_mwh
  )

## Pull in filter for reliability into full_parquet

full_parquet <- full_parquet %>%
  # left_join(system_parquet %>% select(c(
  #   ba_code,
  #   scenario,
  #   re_curtailment = re_curtailment_pct,
  #   non_re_curtailment = non_re_curtailment_pct_of_non_re_gen,
  #   year,
  #   re_limits_dispatch)), by = c("ba_code","scenario","year","re_limits_dispatch")) %>%
  mutate(
    cf_actual = redispatch_mwh / (capacity_mw * (8760+if_else(year %% 4 == 0,24,0)))
  )

## Get ownership data and financials from unit financials data, but only for the assets included in the resource model runs
# Replace data for assets added by resource model and not in unit_level_data with utility data for largest owner in BA


MUL_ownership <- full_parquet %>%
  select(plant_id_eia,generator_id,ba_code) %>% distinct() %>%
  left_join(master_unit_list %>%
              select(c(
                plant_id_eia,
                generator_id,
                fraction_owned,
                Utility_ID,
                entity_type,
                #Technology_FERC,
                td_utility_id_eia,
                entity_type_TD,
                capacity,
                owned_capacity,
                operator_id = utility_id_eia,
                operator_entity_type = entity_type_UT)),
            by = c("plant_id_eia","generator_id")) %>%
  # left_join(full_parquet %>% select(plant_id_eia,generator_id,ba_code,utility_id_eia) %>% filter(!is.na(utility_id_eia)) %>% distinct())
# MUL_ownership_alloc <- allocated_parquet %>%
#   select(plant_id_eia,generator_id,utility_id_eia,ba_code) %>% distinct() %>%
#   left_join(MUL_ownership %>% select(c(plant_id_eia,generator_id,full_utility_id = utility_id_eia)),
#             by = c("plant_id_eia","generator_id")) %>%
#   mutate(utility_id_eia = full_utility_id) %>% select(-c(full_utility_id))
#
# MUL_ownership <- rbind(MUL_ownership,MUL_ownership_alloc) %>% distinct()
#
# MUL_ownership <- MUL_ownership %>%
#   left_join(Utilities, by = c("utility_id_eia"))
#
#
# MUL_ownership <- MUL_ownership %>%
#   left_join(master_unit_list %>%
#               select(c(
#                 plant_id_eia,
#                 generator_id,
#                 fraction_owned,
#                 Utility_ID,
#                 entity_type,
#                 #Technology_FERC,
#                 td_utility_id_eia,
#                 entity_type_TD,
#                 capacity,
#                 owned_capacity,
#                 operator_id = utility_id_eia,
#                 operator_entity_type = entity_type_UT)),
#             by = c("plant_id_eia","generator_id")) %>%
  left_join(Utilities %>% select(c(Utility_ID = utility_id_eia,owner_entity_type_ER = entity_type_ER)), by = ("Utility_ID")) %>%
  left_join(Utilities %>% select(c(td_utility_id_eia = utility_id_eia,td_entity_type_ER = entity_type_ER)), by = ("td_utility_id_eia")) %>%
  mutate(
    entity_type = coalesce(owner_entity_type_ER,entity_type),
    # operator_entity_type = coalesce(entity_type_ER,operator_entity_type),
    entity_type_TD = coalesce(td_entity_type_ER,entity_type_TD)
  ) %>% select(-c(owner_entity_type_ER,td_entity_type_ER))

MUL_ownership <- MUL_ownership %>%
  group_by(Utility_ID,ba_code) %>%
  mutate(ba_owned_capacity = sum(owned_capacity, na.rm = TRUE),
  ) %>% ungroup() %>%
  group_by(operator_id,ba_code) %>%
  mutate(ba_capacity = sum(owned_capacity, na.rm = TRUE),
  ) %>% ungroup() %>%
  group_by(td_utility_id_eia,ba_code) %>%
  mutate(ba_td_capacity = sum(owned_capacity, na.rm = TRUE),
  ) %>% ungroup() %>%
  group_by(ba_code) %>%
  mutate(
    Utility_ID = coalesce(Utility_ID,Utility_ID[which.max(ba_owned_capacity)]),
    td_utility_id_eia = coalesce(td_utility_id_eia,td_utility_id_eia[which.max(ba_td_capacity)]),
    fraction_owned = coalesce(fraction_owned,1),
    entity_type = as.factor(coalesce(entity_type,operator_entity_type,entity_type[which.max(ba_owned_capacity)])),
    entity_type_TD = as.factor(coalesce(entity_type_TD,entity_type_TD[which.max(ba_td_capacity)])),
    #Technology_FERC = as.factor(coalesce(Technology_FERC,Technology_FERC[which.max(ba_owned_capacity)])),
    operator_id = coalesce(operator_id,operator_id[which.max(ba_capacity)]),
    operator_entity_type = as.factor(coalesce(operator_entity_type,operator_entity_type[which.max(ba_capacity)]))#,
    #plant_id_eia = as.numeric(plant_id_eia)
  ) %>% ungroup() %>%
  select(-c(ba_owned_capacity,owned_capacity,capacity,ba_capacity,ba_td_capacity)) %>%
  distinct()

if(!IRA_on) asset_owners <- asset_owners %>% mutate(Direct_Pay = "No", Transferability = "No")

asset_owners <- asset_owners %>%
  mutate(
    public_utility = (entity_type == "F" | entity_type == "M" | entity_type == "P" | entity_type == "S"),
    federal_entity = (entity_type == "F"),
    coop = (entity_type == "C"),
    can_claim_credit = ((!public_utility) | Direct_Pay == "Yes" | Transferability == "Yes"),
    can_monetize_credit = can_claim_credit & (Transferability == "Yes" | Direct_Pay == "Yes"),
    direct_pay_eligible = (Direct_Pay == "Yes"),
    transferability_eligible = (Transferability == "Yes"),
    iou = (entity_type == "I"),
    Technology_FERC = as.factor(Technology_FERC),
    roe = if_else(public_utility,irp_A_rate,roe),
    ror = if_else(public_utility,irp_A_rate,ror),
    equity_ratio = if_else(public_utility,Min_Pub_Equity_Ratio,equity_ratio),
    debt_cost = if_else(public_utility,irp_A_rate,debt_cost)
  )%>% select(-c(Direct_Pay, Transferability))

Disc_Factors <- asset_owners %>%
  select(c(Utility_ID,Technology_FERC,roe,ror)) %>%
  group_by(Utility_ID) %>%
  summarize(
    roe = mean(roe, na.rm = TRUE),
    ror = mean(ror, na.rm = TRUE)
  ) %>%
  cross_join(as.data.frame(all_years), copy = TRUE) %>%
  rename(operating_year = all_years) %>%
  mutate(
    Disc = 1/(1+ror)^(operating_year-irp_year),
    Disc_E = 1/(1+roe)^(operating_year-irp_year)
  ) %>% select(-c(roe,ror))


utilities_inputs_existing <- asset_owners %>%
  select(c(
    Technology_FERC,
    Utility_ID,
    entity_type,
    State_Tax_Rate,
    Blended_Tax_Rate,
    public_utility,
    federal_entity,
    coop,
    iou,
    frac_depr,
    depreciation_rate,
    fixed_OM_frac,
    rem_life,
    ror,
    roe,
    equity_ratio,
    debt_cost,
    capex_per_kw_FERC,
    adjusted_ror
  ))

utilities_inputs_clean <- asset_owners %>%
  select(c(
    Technology_FERC,
    Utility_ID,
    entity_type,
    public_utility,
    federal_entity,
    coop,
    iou,
    can_claim_credit,
    can_monetize_credit,
    direct_pay_eligible,
    transferability_eligible,
    Blended_Tax_Rate,
    PF_Tax_Rate,
    coop_debt,
    coop_equity,
    depreciation_rate,
    fixed_OM_frac,
    Transmission_CAPEX,
    ror,
    roe,
    equity_ratio,
    debt_cost,
    debt_spread,
    adjusted_ror,
  )) %>% distinct()

gc()

## Use averages over ntiles of operational year to fill in original cost and ntiles over actual_cf to fill in maintenance capex

unique_existing_assets <- full_parquet %>%
  filter(Technology_FERC %in% c("steam","other_fossil")) %>%
  filter(!(category %in% c("proposed_clean","proposed_fossil"))) %>%
  select(c(plant_id_eia,generator_id,capacity_mw,capex_per_kw,real_maint_capex_per_kw,Technology_FERC,operational_year,cf_actual,year)) %>%
  group_by(plant_id_eia,generator_id,Technology_FERC,operational_year) %>%
  summarize(
    cf_actual = sum(cf_actual * capacity_mw, na.rm = TRUE) / sum(capacity_mw, na.rm = TRUE),
    capex_per_kw = sum(if_else(is.na(capex_per_kw) | (capex_per_kw<=100),NA,capex_per_kw * capacity_mw), na.rm = TRUE) /
      sum(if_else(is.na(capex_per_kw) | (capex_per_kw<=100),NA,capacity_mw), na.rm = TRUE),
    real_maint_capex_per_kw = sum(if_else(is.na(real_maint_capex_per_kw) | (real_maint_capex_per_kw<=2),NA,real_maint_capex_per_kw * capacity_mw), na.rm = TRUE) /
      sum(if_else(is.na(real_maint_capex_per_kw) | (real_maint_capex_per_kw<=2),NA,capacity_mw), na.rm = TRUE),
    capacity_mw = sum(capacity_mw, na.rm = TRUE) / n_distinct(year)
  ) %>% ungroup() %>%
  distinct() %>%
  group_by(Technology_FERC) %>%
  arrange(operational_year) %>%
  mutate(
    capex_per_kw = if_else(is.na(capex_per_kw) | (capex_per_kw<=100),NA,capex_per_kw),
    w_age_perc_capex = cumsum((!is.na(capex_per_kw)) * capacity_mw)/
      sum((!is.na(capex_per_kw)) * capacity_mw, na.rm = TRUE),
    w_capex_ntile = ceiling(w_age_perc_capex * 20)
  ) %>% ungroup() %>%
  group_by(Technology_FERC) %>%
  arrange(cf_actual) %>%
  mutate(
    real_maint_capex_per_kw = if_else(is.na(real_maint_capex_per_kw) | (real_maint_capex_per_kw<=2),NA,real_maint_capex_per_kw),
    w_cf_perc_maint = cumsum(!is.na(real_maint_capex_per_kw))/
      sum(!is.na(real_maint_capex_per_kw), na.rm = TRUE),
    w_maint_ntile = ntile(w_cf_perc_maint,20)
  ) %>% ungroup() %>%
  group_by(Technology_FERC,w_capex_ntile) %>%
  mutate(capex_per_kw = coalesce(capex_per_kw,median(capex_per_kw, na.rm = TRUE))) %>% ungroup() %>%
  group_by(Technology_FERC,w_maint_ntile) %>%
  mutate(real_maint_capex_per_kw = coalesce(real_maint_capex_per_kw,median(real_maint_capex_per_kw, na.rm = TRUE))) %>% ungroup() %>%
  group_by(plant_id_eia,generator_id) %>% add_count()%>% ungroup()

# Fill in capex and maintenance capex only for existing steam or other_fossil assets (so not for proposed clean or fossil, or for other technologies)

full_parquet <- full_parquet %>%
  select(-c(capex_per_kw,real_maint_capex_per_kw)) %>%
  left_join(unique_existing_assets %>% select(c(plant_id_eia,generator_id,Technology_FERC,capex_per_kw,real_maint_capex_per_kw)) %>% distinct(),
            by = c("plant_id_eia","generator_id", "Technology_FERC"))


# Join ownership data into fossil parquet as well and select only the needed data elements for economic analysis. Note that since this full
# parquet does include proposed fossil (for completeness on emissions calculations), it will have NAs for capex and maint_capex for those assets.

full_parquet_MUL <- full_parquet %>%
  filter(category!="proposed_clean",
         # category!="old_clean",
         # category!="existing_xpatio",
         # (Technology_FERC %in% c("steam","other_fossil"))
         ) %>%
  left_join(MUL_ownership,
            by = c(
              "plant_id_eia" = "plant_id_eia",
              "generator_id" = "generator_id",
              "ba_code" = "ba_code"
            ), relationship = "many-to-many") %>%
  mutate(
    owned_capacity_mw = capacity_mw * fraction_owned,
    Fuel_Costs = coalesce(redispatch_cost_fuel,0) * fraction_owned,
    VOM = coalesce(redispatch_cost_vom,0) * fraction_owned,
    FOM = coalesce(redispatch_cost_fom,0) * fraction_owned,
    Startup_Costs = coalesce(redispatch_cost_startup,0) * fraction_owned,
    owned_opex = Fuel_Costs + VOM + FOM + Startup_Costs,
    MWh_no_curt = redispatch_mwh * fraction_owned,
    owned_redispatch_mwh = redispatch_curt_adj_mwh * fraction_owned,
    #owned_redispatch_mmbtu = redispatch_mmbtu * fraction_owned,
    owned_redispatch_co2_tonne = redispatch_co2_tonne * fraction_owned,
    owned_capex = coalesce(capex_per_kw * 1000 * owned_capacity_mw,NA),
    owned_real_maint_capex = coalesce(real_maint_capex_per_kw,0) * 1000 * owned_capacity_mw
  ) %>% select(c(
    Utility_ID,
    entity_type,
    ba_code,
    scenario,
    plant_id_eia,
    generator_id,
    re_limits_dispatch,
    year,
    #energy_community,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    ccs_scen,
    category,
    technology_description,
    Technology_FERC,
    ccs_eligible,
    owned_redispatch_mwh,
    MWh_no_curt,
    owned_capacity_mw,
    Fuel_Costs,
    VOM,
    FOM,
    Startup_Costs,
    owned_opex,
    #owned_redispatch_mmbtu,
    owned_redispatch_co2_tonne,
    owned_capex,
    owned_real_maint_capex,
    cf_actual,
    #operational_month,
    #retirement_month,
    operational_year,
    retirement_year
  )) %>%
  left_join(utilities_inputs_existing %>%
              select(-c(entity_type)),
            by = c("Utility_ID","Technology_FERC")) %>%
  filter(!is.na(Utility_ID)) %>%
  mutate(
    owned_capex = coalesce(owned_capex, capex_per_kw_FERC * 1000 * owned_capacity_mw),
  )


full_parquet_MUL <- full_parquet_MUL %>%
  group_by(Utility_ID,year,plant_id_eia,generator_id) %>%
  mutate(
    owned_co2_reduced_patio = if_else(owned_redispatch_co2_tonne>0,
                                sum(if_else(scenario=="counterfactual",owned_redispatch_co2_tonne,0), na.rm = TRUE) - owned_redispatch_co2_tonne,0)
  ) %>% ungroup() %>%
  group_by(Utility_ID,plant_id_eia,generator_id) %>%
  mutate(
    owned_co2_reduced = if_else(owned_redispatch_co2_tonne>0,
                                sum(if_else(year==baseline_emissions_year & scenario=="counterfactual",owned_redispatch_co2_tonne,0), na.rm = TRUE) - owned_redispatch_co2_tonne,0)
  ) %>% ungroup()


owned_co2_reductions <- full_parquet_MUL %>%
  # filter(owned_co2_reduced!=0) %>%
  select(c(Utility_ID,year,entity_type,plant_id_eia,generator_id,ba_code,category,coop,scenario,owned_co2_reduced,owned_co2_reduced_patio))

write_parquet(owned_co2_reductions, "owned_co2_reductions.parquet")
#write_parquet(scenario_co2_reductions,"scenario_co2_reductions.parquet")
write_parquet(allocated_parquet, "allocated_parquet.parquet")

full_parquet_MUL <- full_parquet_MUL %>%
  filter((category != "proposed_clean") &
           (category != "proposed_fossil"))

# Fill in data for all years within the IRP NPV range

if(irp_year+NPV_Duration>=2040) {

  last_four_years <- full_parquet_MUL %>%
    filter(year >= 2036)

  extra_years <- c(2040:(irp_year+NPV_Duration))
  repeat_years <- c(2036:2039)
  repeat_years <- as.data.frame(repeat_years)
  repeat_num <- nrow(repeat_years)
  extra_years <- as.data.frame(extra_years) %>%
    cross_join(repeat_years, copy = TRUE) %>%
    filter(repeat_years %% repeat_num == extra_years %% repeat_num) %>%
    left_join(last_four_years, by = c("repeat_years" = "year"), relationship = "many-to-many") %>%
    rename("year" = "extra_years") %>%
    select(-c(repeat_years)) %>%
    filter(is.na(retirement_year) | ((!is.na(retirement_year)) & (year<=retirement_year | retirement_year>2035)))

  full_parquet_MUL <- rbind(full_parquet_MUL, extra_years) %>%
    mutate(
      ba_code = as.factor(ba_code),
      scenario = as.factor(scenario),
      Technology_FERC = as.factor(Technology_FERC)
    )
  rm(extra_years,last_four_years)

} else {

  full_parquet_MUL <- full_parquet_MUL %>%
    filter(year<=irp_year+NPV_Duration)
}


full_parquet_MUL <- full_parquet_MUL %>%
  left_join(CPIU,
            by = c("year" = "Year")) %>%
  group_by(Utility_ID,ba_code,scenario,plant_id_eia,generator_id) %>%
  arrange(year) %>%
  mutate(
    owned_start_depr_reserve = owned_capex * frac_depr,
    # retiring_gen = if_else(is.na(retirement_year),FALSE,(retirement_year<=2035)),
    # rem_life = if_else(retiring_gen,pmax(0,year+1-retirement_year),NA),
    owned_maint_capex = case_when(
      is.na(retirement_year) | ((!is.na(retirement_year)) &
                                  ((year<=retirement_year) | (retirement_year>2035))) ~
        pmax(owned_real_maint_capex * Inflation_Factor_2021,owned_capex * depreciation_rate),
      ((!is.na(retirement_year)) & year<=retirement_year) ~ owned_real_maint_capex * Inflation_Factor_2021,
      TRUE ~ 0
    ),
    owned_capex = owned_capex + cumsum(owned_maint_capex),
    across(c(owned_opex,Fuel_Costs,VOM,FOM,Startup_Costs), ~ . * Inflation_Factor_2021),
    # owned_depr_reserve = owned_start_depr_reserve + cumsum(owned_capex * depreciation_rate),
    # owned_depr_reserve = pmin(owned_capex,owned_depr_reserve),
    # owned_depr_expense = case_when(
    #   (owned_depr_reserve >= owned_capex) &
    #     (owned_depr_reserve - owned_capex * depreciation_rate <=
    #        owned_capex - owned_maint_capex) ~ owned_capex * (1 + depreciation_rate) - owned_depr_reserve,
    #   (owned_depr_reserve >= owned_capex) ~ pmin(owned_capex * depreciation_rate,
    #                                                   owned_maint_capex),
    #   TRUE ~ owned_capex * depreciation_rate
    # ),
    owned_depr_reserve = pmin(owned_capex,owned_start_depr_reserve + cumsum(owned_capex * depreciation_rate)),
    owned_depr_expense = owned_depr_reserve - if_else(year==min(year),owned_start_depr_reserve,lag(owned_depr_reserve,1)),
    owned_depr_reserve = pmin(owned_capex,owned_start_depr_reserve + cumsum(owned_depr_expense)),
    owned_depr_expense = owned_depr_reserve - if_else(year==min(year),owned_start_depr_reserve,lag(owned_depr_reserve,1)),
    owned_depr_reserve = pmin(owned_capex,owned_start_depr_reserve + cumsum(owned_depr_expense)),
    owned_ratebase = owned_capex - (owned_depr_reserve - owned_depr_expense/2),
    frac_depr = 1 - (owned_ratebase/owned_capex),
    depr_life = round(owned_ratebase/owned_depr_expense),
    Average_MWh = if_else(owned_redispatch_mwh>0,mean(if_else(owned_redispatch_mwh>0,owned_redispatch_mwh,NA),na.rm = TRUE),0),
    #depr_life = min(if_else(year==year[which.max(frac_depr)],year-min(year),NA), na.rm = TRUE)
    #test = owned_depr_reserve - (owned_start_depr_reserve + cumsum(owned_depr_expense))
  ) %>% ungroup() %>% select(-c(owned_real_maint_capex,owned_maint_capex,owned_start_depr_reserve))


full_parquet_MUL <- full_parquet_MUL %>%
  mutate(
    adjusted_ror = (roe * equity_ratio) + ((debt_cost * (1 - equity_ratio)) * (1 - Blended_Tax_Rate)),
    Capex_Costs = owned_depr_expense + owned_ratebase * adjusted_ror / (1 - Blended_Tax_Rate),
    Earnings = owned_ratebase * roe * equity_ratio * (1-Blended_Tax_Rate),
    Costs = Capex_Costs + owned_opex
  )

max_years <- c(1:NPV_Duration)

gc()

refinancing_parquet <- full_parquet_MUL %>%
  filter(category=="existing_fossil",
         year %in% build_years,
         year>=irp_year) %>%
  rename("build_year" = "year") %>% filter(re_energy==0) %>%
  left_join(Forward_Interest_Rates %>%
              select(c("Years",
                       "Tenor",
                       "Forward_Treasury_Rates",
                       "DOE_Loan_Rates",
                       "A_Loan_Rates")) %>% filter(Tenor == Debt_Tenor),
            by = c("build_year" = "Years")) %>%
  mutate(
    roe = if_else(public_utility,A_Loan_Rates,roe),
    ror = if_else(public_utility,A_Loan_Rates,ror),
    equity_ratio = if_else(public_utility,Min_Pub_Equity_Ratio,equity_ratio),
    debt_cost = if_else(public_utility,A_Loan_Rates,debt_cost),
    adjusted_ror = (roe * equity_ratio) + ((debt_cost * (1 - equity_ratio)) * (1 - Blended_Tax_Rate)),
    Securitization_Amount = (category == "existing_fossil" & build_year<=EIR_NewERA_fin_build_year) * owned_ratebase * case_when(
      Technology_FERC == "steam" ~ Steam_EIR_Frac,
      Technology_FERC == "other_fossil" ~ Other_Fossil_EIR_Frac,
      TRUE ~ 0
    ),
    Securitization_Rate = case_when(
      coop ~ Forward_Treasury_Rates,
      TRUE ~ DOE_Loan_Rates
    ),
    Securitization_PMT = Securitization_Amount *
      Securitization_Rate / (1-(1/(1+Securitization_Rate))^Debt_Tenor),
    Securitization_OC = (category == "existing_fossil" & build_year<=EIR_NewERA_fin_build_year) * owned_capex * case_when(
      Technology_FERC == "steam" ~ Steam_EIR_Frac,
      Technology_FERC == "other_fossil" ~ Other_Fossil_EIR_Frac,
      TRUE ~ 0
    )
  ) %>% select(-c(
    owned_redispatch_co2_tonne,
    owned_redispatch_mwh,
    MWh_no_curt,
    owned_opex,
    Fuel_Costs,
    VOM,
    FOM,
    Startup_Costs,
    owned_capex,
    owned_co2_reduced,
    re_limits_dispatch,
    nuclear_scen,
    scenario,
    re_energy,
    storage_li_pct,
    ccs_scen,
    ccs_eligible,
    #roe,
    #equity_ratio,
    debt_cost,
    cf_actual,
    operational_year,
    State_Tax_Rate,
    Tenor,
    Forward_Treasury_Rates,
    DOE_Loan_Rates,
    Capex_Costs,
    Costs,
    owned_ratebase,
    Earnings,
    owned_depr_expense,
    owned_depr_reserve,
    rem_life,
    fixed_OM_frac,
    depreciation_rate
  )) %>% distinct() %>% cross_join(as.data.frame(max_years),copy = TRUE) %>%
  rename("cur_project_year" = "max_years") %>%
  filter((cur_project_year <= pmax(depr_life,Debt_Tenor)) & Securitization_Amount>0) %>%
  mutate(
    operating_year = cur_project_year + build_year,
    refi_depr_expense = case_when(
      (cur_project_year<=depr_life) ~ - Securitization_Amount / depr_life,
      TRUE ~ 0
    ),
    refi_depr_reserve = case_when(
      (cur_project_year<=depr_life) ~  - (Securitization_OC - Securitization_Amount) -
        cur_project_year * Securitization_Amount / depr_life,
      (cur_project_year>depr_life) ~ - Securitization_OC,
      TRUE ~ 0
    ),
    refi_capex = - Securitization_OC,
    refi_ratebase = if_else(cur_project_year>depr_life,0,refi_capex - (refi_depr_reserve - refi_depr_expense/2)),
    Earnings = refi_ratebase * roe * equity_ratio,
    Costs = refi_depr_expense + refi_ratebase * adjusted_ror / (1 - Blended_Tax_Rate) +
      if_else((cur_project_year>0) & (cur_project_year<=Debt_Tenor),Securitization_PMT,0),
    Capex_Costs = Costs,
    Opex = 0
  ) %>% select(c(
    Utility_ID,
    entity_type,
    plant_id_eia,
    generator_id,
    ba_code,
    build_year,
    operating_year,
    cur_project_year,
    technology_description,
    Technology_FERC,
    depr_life,
    refi_depr_expense,
    refi_capex,
    refi_ratebase,
    refi_depr_reserve,
    Earnings,
    Costs,
    Opex,
    Capex_Costs,
    Securitization_Amount,
    Securitization_Rate,
    Securitization_PMT,
    Securitization_OC
  ))

refinancing_parquet <- refinancing_parquet %>%
  group_by(Utility_ID,entity_type,plant_id_eia,generator_id,ba_code,build_year) %>%
  arrange(cur_project_year) %>%
  mutate(
    Forward_Earnings = lead(Earnings, n = 3)
  ) %>% ungroup()

fossil_parquet <- full_parquet_MUL %>%
  rename(
    operating_year = year,
    MW = owned_capacity_mw,
    MWh = owned_redispatch_mwh,
    Emissions = owned_redispatch_co2_tonne,
    Emissions_Reduced = owned_co2_reduced,
    Capex = owned_capex,
    Opex = owned_opex
  ) %>%
  group_by(Utility_ID,entity_type,plant_id_eia,generator_id,ba_code,category,scenario) %>%
  arrange(desc(operating_year)) %>%
  mutate(
    build_year = operating_year-1,
    Average_Emissions = mean(Emissions),
    Average_Emissions_Reductions = mean(Emissions_Reduced),
    Forward_Earnings = lead(Earnings, n = 3)
  ) %>% ungroup()

gc()

rm(
  full_parquet,
  full_parquet_map,
  full_parquet_MUL,
  full_proposed_parquet,
  # sales_ult_cust_861,
  # sales_ult_cust_861_total,
  utilities_eia860,
  allocated_parquet
  )

write_parquet(refinancing_parquet, "refinancing.parquet")
write_parquet(fossil_parquet,"fossil.parquet")
gc()
fossil_parquet <- read_parquet("fossil.parquet")
refinancing_parquet <- read_parquet("refinancing.parquet")

final_fossil_costs_by_utility_ba <- fossil_parquet %>%
  filter(!is.na(Costs),scenario != "historical") %>%
  rename(Utility_ID_Econ = Utility_ID) %>%
  group_by(ba_code,
           Utility_ID_Econ,
           scenario,
           category,
           re_energy,
           nuclear_scen,
           storage_li_pct,
           ccs_scen,
           re_limits_dispatch,
           Technology_FERC,
           technology_description,
           operating_year) %>%
  summarize(
    across(c(Average_Emissions,Average_Emissions_Reductions,Emissions,Emissions_Reduced,MW,MWh,MWh_no_curt,
             Fuel_Costs,VOM,FOM,Startup_Costs,Capex,Capex_Costs,Opex,Costs,Earnings,Forward_Earnings), ~ sum(., na.rm = TRUE))
  ) %>% ungroup()

gc()

final_refinancing_costs_by_utility_ba <- refinancing_parquet %>%
  filter(!is.na(Costs)) %>%
  rename(Utility_ID_Econ = Utility_ID) %>%
  group_by(ba_code,
           Utility_ID_Econ,
           build_year,
           cur_project_year,
           Technology_FERC,
           technology_description,
           operating_year) %>%
  summarize(
    across(c(Securitization_Amount,Costs,Earnings,Forward_Earnings), ~ sum(., na.rm = TRUE))
  ) %>% ungroup() %>%
  mutate(
    MW = 0,
    MWh = 0,
    MWh_no_curt = 0,
    Fuel_Costs = 0,
    VOM = 0,
    FOM = 0,
    Startup_Costs = 0,
    Capex_Costs = Costs,
    Capex = 0,
    Opex = 0,
    category = "refinancing",
    Emissions = 0,
    Emissions_Reduced = 0,
    NREL_Class = NA
  )

write_parquet(final_fossil_costs_by_utility_ba,"final_fossil_costs_by_utility_ba.parquet")
write_parquet(final_refinancing_costs_by_utility_ba, "final_refinancing_costs_by_utility_ba.parquet")

## Clean up to free up memory!

# rm(fossil_parquet, full_parquet_MUL, owned_co2_reductions, allocated_parquet, refinancing_parquet)
gc()

## Summarize renewable parquet by owner, renewable asset, scenario, and year - and calculate ownership
## weighted transmission and storage

allocated_parquet <- read_parquet("allocated_parquet.parquet")

allocated_parquet_MUL <- allocated_parquet %>%
  filter(category!="patio_transmission_upgrades") %>%
  left_join(MUL_ownership,relationship = "many-to-many",
            by = c(
              "plant_id_eia" = "plant_id_eia",
              "generator_id" = "generator_id",
              "ba_code" = "ba_code"
            )) %>%
  mutate(Utility_ID_Econ = Utility_ID) %>%
  group_by(
    Utility_ID,
    Utility_ID_Econ,
    Technology_FERC,
    entity_type,
    ba_code,
    scenario,
    re_plant_id,
    re_generator_id,
    plant_id_eia,
    generator_id,
    re_limits_dispatch,
    year,
    energy_community,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    ccs_scen,
    category,
    technology_description,
    operational_year,
    ccs_eligible,
    NREL_Class
  ) %>%
  summarize(
    across(c(capacity_mw,redispatch_mwh,redispatch_mmbtu,redispatch_mmbtu,redispatch_co2_tonne,redispatch_cost_fuel,
           co2_reduced,redispatch_cost_startup,implied_need_mw), ~ sum(. * fraction_owned, na.rm = TRUE),.names = "owned_{.col}"),
    MWh_no_curt = sum(fraction_owned * MWh_no_curt, na.rm = TRUE),
    operational_year = max(operational_year),
    cf_actual = sum(cf_actual * fraction_owned * redispatch_mwh, na.rm = TRUE) / owned_redispatch_mwh
  ) %>% ungroup()

## Flag here that transmission line and upgrade ownership should be based on FERC data on ownership of the current
## system by ba rather than generation ownership.

transmission_lines_parquet <- allocated_parquet %>%
  filter(category=="patio_clean") %>%
  left_join(MUL_ownership %>% select(
    plant_id_eia,
    generator_id,
    ba_code,
    Utility_ID_Econ = Utility_ID,
    Utility_ID = td_utility_id_eia,
    entity_type = entity_type_TD,
    fraction_owned) %>% distinct(),relationship = "many-to-many",by = c("plant_id_eia","generator_id","ba_code")) %>%
  group_by(
    Utility_ID,
    Utility_ID_Econ,
    entity_type,
    ba_code,
    scenario,
    re_plant_id,
    re_generator_id,
    plant_id_eia,
    generator_id,
    re_limits_dispatch,
    year,
    #energy_community,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    ccs_scen,
    category,
    #technology_description,
    #ccs_eligible,
    #NREL_Class
  ) %>%
  summarize(owned_capacity_mw = sum(fraction_owned * distance * capacity_mw * 0.621371, na.rm = TRUE)) %>% ungroup() %>%
  mutate(
    #re_plant_id = NA,
    category = as.factor("transmission_lines"),
    Technology_FERC = as.factor("transmission"),
    technology_description = as.factor("Transmission"),
    category = as.factor("transmission_lines"),
    ccs_eligible = FALSE,
    NREL_Class = NA,
    owned_redispatch_mwh = 0,
    owned_redispatch_mmbtu = 0,
    owned_redispatch_co2_tonne = 0,
    owned_redispatch_cost_fuel = 0,
    owned_redispatch_cost_startup = 0,
    owned_co2_reduced = 0,
    #owned_stored_co2_tonne = 0,
    owned_implied_need_mw = 0,
    MWh_no_curt = 0,
    operational_year = NA,
    cf_actual = NA,
    energy_community = FALSE
  )

proposed_transmission_upgrades_parquet <- allocated_parquet %>%
  filter(category=="proposed_clean" | category == "proposed_fossil") %>%
  mutate(category = "proposed_transmission_upgrades") %>%
  left_join(MUL_ownership %>% select(
    plant_id_eia,
    generator_id,
    ba_code,
    Utility_ID_Econ = Utility_ID,
    Utility_ID = td_utility_id_eia,
    entity_type = entity_type_TD,
    fraction_owned) %>% distinct(),relationship = "many-to-many",by = c("plant_id_eia","generator_id","ba_code")) %>%
  group_by(
    Utility_ID,
    Utility_ID_Econ,
    entity_type,
    ba_code,
    scenario,
    plant_id_eia,
    generator_id,
    #re_plant_id,
    re_limits_dispatch,
    year,
    #energy_community,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    ccs_scen,
    operational_year,
    category,
    #technology_description,
    #ccs_eligible,
    #NREL_Class
  ) %>%
  summarize(owned_capacity_mw = sum(capacity_mw * fraction_owned, na.rm = TRUE)) %>% ungroup() %>%
  mutate(
    re_plant_id = paste(Utility_ID,Utility_ID_Econ,ba_code,category,plant_id_eia,generator_id,sep = "_"),
    re_generator_id = generator_id,
    Technology_FERC = "transmission",
    technology_description = as.factor("Transmission"),
    ccs_eligible = FALSE,
    NREL_Class = NA,
    owned_redispatch_mwh = 0,
    owned_redispatch_mmbtu = 0,
    owned_redispatch_co2_tonne = 0,
    owned_redispatch_cost_fuel = 0,
    owned_redispatch_cost_startup = 0,
    owned_co2_reduced = 0,
    #owned_stored_co2_tonne = 0,
    owned_implied_need_mw = 0,
    MWh_no_curt = 0,
    cf_actual = NA,
    energy_community = FALSE
  )

patio_transmission_upgrades_parquet <- allocated_parquet %>%
  filter(category=="patio_transmission_upgrades") %>%
  left_join(MUL_ownership %>% select(
    plant_id_eia,
    generator_id,
    ba_code,
    Utility_ID_Econ = Utility_ID,
    Utility_ID = td_utility_id_eia,
    entity_type = entity_type_TD,
    fraction_owned) %>% distinct(),relationship = "many-to-many",by = c("plant_id_eia","generator_id","ba_code")) %>%
  group_by(
    Utility_ID,
    Utility_ID_Econ,
    entity_type,
    ba_code,
    scenario,
    #re_plant_id,
    plant_id_eia,
    generator_id,
    re_limits_dispatch,
    year,
    #energy_community,
    re_energy,
    nuclear_scen,
    storage_li_pct,
    ccs_scen,
    #operational_year,
    category,
    #technology_description,
    #ccs_eligible,
    #NREL_Class
  ) %>%
  summarize(owned_capacity_mw = sum(capacity_mw * fraction_owned, na.rm = TRUE)) %>% ungroup() %>%
  mutate(
    re_plant_id = paste(Utility_ID,Utility_ID_Econ,ba_code,category,plant_id_eia,generator_id,sep = "_"),
    re_generator_id = generator_id,
    Technology_FERC = "transmission",
    technology_description = as.factor("Transmission"),
    ccs_eligible = FALSE,
    NREL_Class = NA,
    owned_redispatch_mwh = 0,
    owned_redispatch_mmbtu = 0,
    owned_redispatch_co2_tonne = 0,
    owned_redispatch_cost_fuel = 0,
    owned_redispatch_cost_startup = 0,
    owned_co2_reduced = 0,
    #owned_stored_co2_tonne = 0,
    owned_implied_need_mw = 0,
    MWh_no_curt = 0,
    cf_actual = NA,
    operational_year = NA,
    energy_community = FALSE
  )


allocated_parquet_MUL <- rbind(allocated_parquet_MUL,
                               transmission_lines_parquet,
                               proposed_transmission_upgrades_parquet,
                               patio_transmission_upgrades_parquet)

allocated_parquet_MUL <- allocated_parquet_MUL %>%
  mutate(
    ba_code = as.factor(ba_code),
    re_plant_id = as.factor(re_plant_id),
    scenario = as.factor(scenario),
    category = as.factor(category),
    technology_description = as.factor(technology_description),
    Technology_FERC = as.factor(Technology_FERC)
  )

# Step 1b: Prep tidy dataframe for new renewables, storage, and fossil

## Add in years that are repeated on a four-year cycle post-2035

if(irp_year+NPV_Duration>=2040) {

  last_four_years <- allocated_parquet_MUL %>%
    filter(year >= 2036)

  extra_years <- c(2040:(irp_year+NPV_Duration))
  repeat_years <- c(2036:2039)
  repeat_years <- as.data.frame(repeat_years)
  repeat_num <- nrow(repeat_years)
  extra_years <- as.data.frame(extra_years) %>%
    cross_join(repeat_years, copy = TRUE) %>%
    filter(repeat_years %% repeat_num == extra_years %% repeat_num) %>%
    left_join(last_four_years, by = c("repeat_years" = "year"), relationship = "many-to-many") %>%
    rename("year" = "extra_years") %>%
    select(-c(repeat_years))

  consolidated_final_data_inputs <- rbind(allocated_parquet_MUL, extra_years)
  rm(extra_years,
     repeat_years,
     last_four_years)

} else {
  consolidated_final_data_inputs <- allocated_parquet_MUL %>%
    filter(year<=irp_year+NPV_Duration)
}

consolidated_final_data_inputs <- consolidated_final_data_inputs %>%
  filter(owned_capacity_mw>0) %>%
  rename(
    operating_year = year,
    MW = owned_capacity_mw,
    MWh = owned_redispatch_mwh
  ) %>%
  mutate(
    technology_description = as.factor(technology_description),
    counterfactual = (scenario=="counterfactual"),
    energy_community = coalesce(energy_community,FALSE),
    proposed = (category %in% c("proposed_fossil","proposed_clean","proposed_transmission_upgrades"))
  ) %>% filter(!is.na(technology_description))

write_parquet(consolidated_final_data_inputs,"consolidated_final_data_inputs.parquet")

rm(allocated_parquet_MUL,
   # allocated_parquet,
   # full_proposed_parquet,
   transmission_lines_parquet,
   transmission_upgrades_patio_parquet,
   patio_transmission_upgrades_parquet,
   proposed_transmission_upgrades_parquet)

### Start here for economic analysis and scenario selection if data prep above has been run at least once in current R-session (only need to run through line 1069)

consolidated_final_data_inputs <- read_parquet("consolidated_final_data_inputs.parquet")
fossil_parquet <- read_parquet("fossil.parquet")
refinancing_parquet <- read_parquet("refinancing.parquet")

final_fossil_costs_by_utility_ba <- read_parquet("final_fossil_costs_by_utility_ba.parquet")
# final_refinancing_costs_by_utility_ba <- read_parquet("final_refinancing_costs_by_utility_ba.parquet")

## First, pull in some important constants

if (IRA_on == FALSE) {
  Tax_Credits <- Tax_Credits %>%
    select(-c(PTC, ITC)) %>%
    rename(PTC = PTC_Cur,
           ITC = ITC_Cur)
}

Inflator_2022 = (CPIU %>% subset(Year==2022,c(Inflation_Factor_2021)) %>% as.numeric)
Inflator_2025 = (CPIU %>% subset(Year==2025,c(Inflation_Factor_2021)) %>% as.numeric)
NewERA_subsidy_cost <- as.numeric(NewERA_Credit_Subsidy_Table %>%
                                    filter(NewERA_Interest_Rate == NewERA_Rate_Chosen) %>%
                                    select(credit_subsidy_cost))

MACRS_long <- MACRS %>%
  pivot_longer(cols = c(LandbasedWind,
                        UtilityPV,
                        OffShoreWind,
                        `Utility-Scale Battery Storage`,
                        Nuclear,
                        Transmission,
                        Coal_FE,
                        NaturalGas_FE,
                        Coal_Retrofits,
                        NaturalGas_Retrofits),
               names_to = "technology_description",
               values_to = "cur_MACRS") %>%
  group_by(technology_description) %>%
  arrange(cur_project_year) %>%
  mutate(sum_MACRS=cumsum(cur_MACRS)) %>% ungroup()

owned_co2_reductions <- read_parquet("owned_co2_reductions.parquet")

## Pull in the build years and create dataframe that has row for each owner / asset class that has either already been
## proposed in EIA-860 or is proposed to be build by the Patio model, for each scenario, and for each year that
## the asset would be operational. This will be used to pull in all external data input parameters.

data_input_parameters <- consolidated_final_data_inputs %>%
  cross_join(as.data.frame(build_years), copy = TRUE) %>%
  rename("build_year" = "build_years") %>%
  filter(operating_year > build_year,
         (proposed & (build_year == operational_year-1)) |
           ((re_energy<=(build_year-(irp_year-1))/5) & !proposed) ) %>%
  select(c(
    build_year,
    Utility_ID,
    Utility_ID_Econ,
    Technology_FERC,
    entity_type,
    operating_year,
    energy_community,
    category,
    proposed,
    technology_description,
    NREL_Class
  )) %>% distinct()

gc()

## Pre-create several indices that we will use in the economic model loop to reduce calculation times in group_by calls.
# The grouping can be as simple as tagging all assets owned by a particular utility in a given operating year to
# more granular groupings of lines corresponding to specific assets in a given category and technology
# over all operating years.

final_data_index <- consolidated_final_data_inputs %>%
  cross_join(as.data.frame(build_years), copy = TRUE) %>%
  rename("build_year" = "build_years") %>%
  filter(operating_year > build_year,
         (proposed & (build_year == operational_year-1)) |
           ((re_energy<=(build_year-(irp_year-1))/5) & !proposed) ) %>%
  select(c(
    build_year,
    Utility_ID,
    Utility_ID_Econ,
    Technology_FERC,
    entity_type,
    ba_code,
    scenario,
    re_plant_id,
    re_generator_id,
    plant_id_eia,
    generator_id,
    # re_limits_dispatch,
    energy_community,
    # re_energy,
    # nuclear_scen,
    # storage_li_pct,
    # ccs_scen,
    category,
    proposed,
    technology_description,
    ccs_eligible,
    # operating_year,
    NREL_Class
  )) %>% distinct() %>%
  filter(!is.na(build_year)) %>%
  mutate(index=row_number()) # %>%
  # group_by(
  #   build_year,
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   entity_type,
  #   ba_code,
  #   scenario,
  #   re_plant_id,
  #   re_generator_id,
  #   plant_id_eia,
  #   generator_id,
  #   energy_community,
  #   category,
  #   proposed,
  #   technology_description,
  #   ccs_eligible,
  #   NREL_Class
  # ) %>% mutate(index = max(op_index)) %>% ungroup() %>%
  # group_by(
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   entity_type,
  #   ba_code,
  #   re_plant_id,
  #   re_generator_id,
  #   plant_id_eia,
  #   generator_id,
  #   energy_community,
  #   category,
  #   proposed,
  #   technology_description,
  #   ccs_eligible,
  #   operating_year,
  #   NREL_Class
  # ) %>% mutate(asset_year_index = max(op_index)) %>% ungroup() %>%
  # group_by(
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   build_year,
  #   entity_type,
  #   ba_code,
  #   re_plant_id,
  #   re_generator_id,
  #   plant_id_eia,
  #   generator_id,
  #   energy_community,
  #   category,
  #   proposed,
  #   technology_description,
  #   ccs_eligible,
  #   NREL_Class
  # ) %>% mutate(asset_vintage_index = max(op_index)) %>% ungroup() %>%
  # group_by(
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   entity_type,
  #   ba_code,
  #   re_plant_id,
  #   re_generator_id,
  #   plant_id_eia,
  #   generator_id,
  #   energy_community,
  #   category,
  #   proposed,
  #   technology_description,
  #   ccs_eligible,
  #   NREL_Class
  # ) %>% mutate(asset_index = max(op_index)) %>% ungroup() %>%
  # group_by(
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   ba_code,
  #   scenario,
  #   technology_description,
  #   operating_year,
  #   NREL_Class
  # ) %>% mutate(asset_type_scen_opyear_index = max(op_index)) %>% ungroup() %>%
  # group_by(
  #   Utility_ID,
  #   Utility_ID_Econ,
  #   Technology_FERC,
  #   ba_code,
  #   technology_description,
  #   operating_year,
  #   NREL_Class
  # ) %>% mutate(asset_type_opyear_index = max(op_index)) %>% ungroup()

write_parquet(final_data_index, "final_data_index.parquet")

gc()

## Populate comprehensive data inputs frame that incorporates NREL inputs, inflators, forward interest rates,
## and cost of capital appropriately for each possible technology / build and operating year combination.
## Pull in tax incentive information for all assets.

data_input_parameters <- data_input_parameters %>%
  left_join(CPIU, by = c("build_year" = "Year")) %>%
  left_join(Tax_Credits %>%
              select(c(Technology,
                       ITC,
                       PTC,
                       Domestic_Content_Percent,
                       supply_chain_tariff_CAPEX_impact,
                       Year)),
            by = c("build_year" = "Year",
                   "technology_description" = "Technology")) %>%
  rename(Build_Year_Inflator = Inflation_Factor_2021) %>%
  left_join(nrel_atb_tech_intermediate,
            by = c("technology_description" = "technology",
                   "NREL_Class" = "techdetail",
                   "build_year" = "core_metric_variable")) %>%
  rename("CCS_Parasitic_Load" = "Heat Rate Penalty") %>%
  left_join(Tax_Equity_Params %>% select(c(
    Technology,
    Normalized,
    ITC_Norm_Period,
    Acc_Norm_Public_Coop,
    CCS_CO2_Frac,
    Earliest_Build_Year
  )), by = c("technology_description" = "Technology")) %>%
  left_join(CPIU, by = c("operating_year" = "Year")) %>%
  left_join(Forward_Interest_Rates %>%
              select(c("Years",
                       "Tenor",
                       "Forward_Treasury_Rates",
                       "DOE_Loan_Rates",
                       "BBB_Loan_Rates",
                       "B_Loan_Rates",
                       "BB_Loan_Rates",
                       "A_Loan_Rates",
                       "AA_Loan_Rates",
                       "AAA_Loan_Rates"
              )) %>% filter(Tenor == Clean_Life),
            by = c("build_year" = "Years")) %>%
  left_join(utilities_inputs_clean %>%
              select(-c(entity_type)),
            by = c("Utility_ID","Technology_FERC")) %>%
  mutate(cur_project_year = operating_year - build_year) %>%
  left_join(MACRS_long,by = c("technology_description","cur_project_year"))

## Fill in any missing data with averages or default values in data input parameters, update key financial metrics
## to reflect future interest rate expectations, fill in transmission costs from historic regressions / data,
## set discount factors for NPV calculation, and define additional ITC / PTC dummy variables

if(!IRA_on) data_input_parameters <- data_input_parameters %>% mutate(Normalized = "Yes")

data_input_parameters <- data_input_parameters %>%
  mutate(
    tax_equity_used = (direct_pay_eligible) & (!iou) & (Technology_FERC!="transmission"),
    tax_equity_used = coalesce(tax_equity_used, TRUE),
    Normalized = coalesce(as.factor(Normalized), as.factor("Yes")),
    CAPEX = coalesce(CAPEX,if_else(category=="transmission_lines",Transmission_CAPEX,trans_costs/1000)) * Build_Year_Inflator,
    `Fixed O&M` = if_else(technology_description=="Transmission",
                          CAPEX * fixed_OM_frac,
                          coalesce(`Fixed O&M` * Build_Year_Inflator, CAPEX * fixed_OM_frac)),
    Asset_Life = if_else(technology_description=="Transmission",
                         pmax(if_else(depreciation_rate>0,round(1/depreciation_rate),0),Clean_Life),
                         Clean_Life),
    `Variable O&M` = coalesce(`Variable O&M`,0) * Build_Year_Inflator,
    Inflator = Inflation_Factor_2021 / Build_Year_Inflator,
    PTC_Asset = (PTC>0),
    PTC_Year = (operating_year<=build_year+PTC_Duration) & PTC_Asset,
    Fuel = coalesce(Fuel, 0),
    roe = if_else(public_utility,A_Loan_Rates,roe),
    debt_cost = if_else(public_utility & build_year<=EIR_NewERA_fin_build_year,
                        (A_Loan_Rates) * (1-Public_EIR_Frac)+
                          DOE_Loan_Rates * Public_EIR_Frac,
                        debt_cost),
    ror = if_else(public_utility,A_Loan_Rates,ror),
    equity_ratio = if_else(public_utility,Min_Pub_Equity_Ratio,equity_ratio),
    Disc_Fin = 1/(1+ror)^(operating_year-build_year),
    ITC_Norm_Period = case_when(
      (Normalized == "Yes") & iou ~ Asset_Life,
      (Acc_Norm_Public_Coop == "Yes") & (public_utility | coop) ~ ITC_Norm_Period,
      TRUE ~ Asset_Life
    ),
    ITC_Normalized = (iou & (Normalized =="Yes")),
    ITC = (can_claim_credit) * (ITC + 0.1 * (Domestic_Content_Percent + energy_community)),
    ITC_claim = ITC * (cur_project_year <= ITC_Norm_Period),
    ITC_reg = ITC_claim * if_else(transferability_eligible,Transferability_Discount,1),
    Clean_PTC_est = coalesce(- can_claim_credit * PTC_Year * CF *
      (8760 + (operating_year %% 4 == 0)*24) *
      (1 + (Domestic_Content_Percent + energy_community) * 0.1) *
      round(2 * PTC * PTC_Inflator_2022 * PTC_Value * Inflation_Factor_2021 / Inflator_2022) *
      0.5 * if_else(transferability_eligible,Transferability_Discount,1),0),
    Clean_CapEx_est = (1 + supply_chain_tariff_CAPEX_impact) * CAPEX * 1000
  ) %>%
  select(-c(Transmission_CAPEX)) %>%
  filter(proposed | (cur_project_year<=Asset_Life) & !proposed)

data_input_parameters <- data_input_parameters %>%
  group_by(Utility_ID,Utility_ID_Econ,build_year,energy_community,technology_description,NREL_Class,category,proposed) %>%
  mutate(
    ITC_Better = (-sum(Clean_PTC_est * Disc_Fin,na.rm = TRUE) <  ITC_reg * Clean_CapEx_est) & (ITC_reg>0)
  ) %>% ungroup()

gc()

### Run economic scenario analysis from here to replace existing economic model results with new set of scenario outputs

replace_results = TRUE
test_loop = FALSE
ba_scenario_selection = TRUE
utility_override_for_counterfactual_ba = FALSE
utility_savings_overshoot = 1.025

refinancing_capital_recycling_trigger = 2.0
ba_savings_tolerance = 0.5
ba_earnings_threshold = 0.5
ba_least_cost = FALSE
New_Asset_Debt_Replaced_by_EIR = 0.3
New_Asset_Equity_Replaced_by_EIR = 0.2

refinancing_capital_recycling_trigger_range <- c(1.5)
ba_savings_tolerance_range <- c(0.5)
ba_earnings_threshold_range <- c(0.5)
ba_least_cost_range <- c(TRUE,FALSE)
New_Asset_Debt_Replaced_by_EIR_Range <- c(0.2)
New_Asset_Equity_Replaced_by_EIR_Range <- c(0.2)

# clear space
rm(
  allocated_parquet,
  asset_owners,
  full_parquet,
  full_parquet_map,
  full_proposed_parquet,
  master_unit_list,
  MUL_ownership,
  nrel_atb,
  patio_cap_changes,
  renewable_nrel_class,
  Utilities,
  utilities_eia860,
  utilities_inputs_clean,
  utilities_inputs_existing
)

gc()

if(replace_results){
  file.remove("final_outputs.parquet")
  file.remove("scenario_selected.parquet")
}

for(refinancing_capital_recycling_trigger in refinancing_capital_recycling_trigger_range){
for(ba_savings_tolerance in ba_savings_tolerance_range){
for(ba_earnings_threshold in ba_earnings_threshold_range){
for(ba_least_cost in ba_least_cost_range){
for(New_Asset_Debt_Replaced_by_EIR in New_Asset_Debt_Replaced_by_EIR_Range){
for(New_Asset_Equity_Replaced_by_EIR in New_Asset_Equity_Replaced_by_EIR_Range){


if (test_loop) {
  loop_years <- c((irp_year):(irp_year))
} else {
  loop_years <- c(irp_year:final_loop_year)
}

gc()

# This initializes the dataframe that records which scenario is selected as the model loops through build years. Use test mode to
# iterate through the loop manually.

early_years <- c(baseline_emissions_year:(min(loop_years)-1))

all_utility_ba_codes <- rbind(consolidated_final_data_inputs %>% select(c(Utility_ID_Econ,ba_code)) %>% distinct(),
                              fossil_parquet %>% select(c(Utility_ID_Econ = Utility_ID,ba_code)) %>% distinct()) %>% distinct()

all_scenarios <- all_utility_ba_codes %>%
  cross_join(system_parquet %>% select(c(scenario,re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)) %>%
               filter(re_energy>=0) %>% distinct(), copy = TRUE) %>%
  mutate(scenario = as.factor(scenario)) %>%
  inner_join(valid_ba_scens, by = c("ba_code","scenario"))

scenario_selected <- all_scenarios %>%
  cross_join(as.data.frame(early_years), copy = TRUE) %>% rename(build_year = early_years) %>%
  filter(re_energy<=pmax(0,(build_year-(irp_year-1))/5)) %>%
  mutate(
    is_selected = as.logical(
      # ((Utility_ID_Econ %% 2 == 0) & (re_energy==0.2) & (storage_li_pct == 0.25) & (build_year == irp_year) & test_loop) |
        # ((Utility_ID_Econ %% 2 == 1) & (re_energy==0) & (build_year == irp_year) & test_loop) |
        ( (re_energy==0) & (build_year != irp_year)) )) %>%
  group_by(Utility_ID_Econ,ba_code) %>%
  arrange(is_selected,build_year) %>%
  mutate(
    re_energy_max = pmax(0,max(re_energy * is_selected)),
    storage_li_pct_max = pmax(0,max(storage_li_pct * is_selected)),
    re_still_limits_dispatch = as.logical(sum(if_else(is_selected,re_limits_dispatch,NA), na.rm = TRUE)>0),
    nuclear_scen_max = pmax(0,max(nuclear_scen * is_selected)),
    ccs_scen_max = pmax(0,max(ccs_scen * is_selected)),
    scenario = as.factor(scenario)
    # sec_change = if_else(is_selected,
    #                      pmax(re_energy,nuclear_scen,ccs_scen) -
    #                        pmax(0,lag(re_energy,n=1),lag(nuclear_scen,n=1),lag(ccs_scen,n=1), na.rm = TRUE),0)
  ) %>% ungroup() # %>% select(-c(re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen))

if (test_loop) {
  loop_years <- c((irp_year):(irp_year))
} else {
  loop_years <- c(irp_year:final_loop_year)
}

gc()

# Economic Analysis Loop

for(cur_build_year in loop_years) {

  cur_loop_years <- c(min(loop_years):cur_build_year)
  cur_build_years <- c(min(early_years):cur_build_year)

  cur_scenario_selected <- scenario_selected %>%
    filter(build_year == (cur_build_year-1))

  scenario_selected_operating <- scenario_selected %>%
    mutate(operating_year = build_year+1) %>% select(-c(build_year))

  selectable_scenarios <- all_scenarios %>%
    left_join(cur_scenario_selected %>%
                select(c(Utility_ID_Econ,ba_code,re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max)) %>%
                distinct(), by = c("Utility_ID_Econ","ba_code")) %>%
    mutate(
      across(c(re_energy_max,storage_li_pct_max,nuclear_scen_max,ccs_scen_max), ~ coalesce(.x,0)),
      re_still_limits_dispatch = coalesce(re_still_limits_dispatch,TRUE),
    ) %>%
    filter(re_energy<=(cur_build_year-(irp_year-1))/5,
           re_energy>=re_energy_max,
           storage_li_pct>=storage_li_pct_max,
           (re_still_limits_dispatch == re_limits_dispatch) | !re_limits_dispatch,
           nuclear_scen>=nuclear_scen_max,
           ccs_scen>=ccs_scen_max) %>%
    mutate(
      # sec_change = pmax(re_energy,storage_li_pct,nuclear_scen,ccs_scen, na.rm = TRUE) - pmax(re_energy_max,storage_li_pct,nuclear_scen_max,ccs_scen_max, na.rm = TRUE),
      build_year = cur_build_year,
      scenario = as.factor(scenario),
      is_selected = FALSE
    ) # %>% select(-c(re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max))

  all_scenarios_build <- rbind(scenario_selected,selectable_scenarios) %>%
    select(-c(re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max))
  all_scenarios_operating <- all_scenarios_build %>%
    mutate(operating_year = build_year+1) %>% select(-c(build_year))

  # Join data_input parameters for current and all previous build years, filter out all scenarios from previous years that were not
  # selected, and pull in the unique asset / scenario index for simplifying calculations.

  gc()

  final_data_inputs_selected_years <- selectable_scenarios %>%
    select(c(Utility_ID_Econ,ba_code,scenario,
             re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)) %>%
    left_join(all_scenarios_operating %>%
                filter(is_selected) %>%
                inner_join(consolidated_final_data_inputs %>%
                             select(-c(re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)),
                           by = c(
                             "Utility_ID_Econ",
                             "ba_code",
                             "scenario",
                             "operating_year"
                           )) %>%
                select(-c(scenario,re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)), by = c(
                  "Utility_ID_Econ",
                  "ba_code"
                ), relationship = "many-to-many")

  final_data_inputs <- selectable_scenarios %>%
    select(c(Utility_ID_Econ,ba_code,scenario,is_selected)) %>%
    inner_join(consolidated_final_data_inputs %>%
                 filter(operating_year>cur_build_year), by = c(
                   "Utility_ID_Econ",
                   "ba_code",
                   "scenario"
                 ))

  final_data_inputs <- rbind(final_data_inputs,final_data_inputs_selected_years) %>%
    group_by(Utility_ID_Econ,Utility_ID,ba_code,re_plant_id,re_generator_id,plant_id_eia,generator_id,technology_description) %>%
    mutate(
      build_year = case_when(
        proposed ~ operational_year-1,
        TRUE ~ min(operating_year)-1
      )
    ) %>% ungroup() %>%
    left_join(final_data_index, by = c(
      "build_year",
      "Utility_ID",
      "Utility_ID_Econ",
      "Technology_FERC",
      "entity_type",
      "ba_code",
      "scenario",
      "re_plant_id",
      "re_generator_id",
      "plant_id_eia",
      "generator_id",
      # "re_limits_dispatch",
      "energy_community",
      # "re_energy",
      # "nuclear_scen",
      # "storage_li_pct",
      # "ccs_scen",
      "category",
      "proposed",
      "technology_description",
      "ccs_eligible",
      # "operating_year",
      "NREL_Class"
    )) %>%
    mutate(
      scenario = as.factor(scenario),
      technology_description = as.factor(technology_description)
    )

  gc()

  # Reflect the selected scenarios in the fossil costs

  fossil_selected_years <- selectable_scenarios %>%
    select(c(Utility_ID_Econ,ba_code,scenario,
             re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)) %>%
    left_join(all_scenarios_operating %>%
                filter(is_selected) %>%
                inner_join(final_fossil_costs_by_utility_ba %>%
                             select(-c(re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)),
                           by = c(
                             "Utility_ID_Econ",
                             "ba_code",
                             "scenario",
                             "operating_year"
                           )) %>%
                select(-c(scenario,re_energy,storage_li_pct,re_limits_dispatch,nuclear_scen,ccs_scen)), by = c(
                  "Utility_ID_Econ",
                  "ba_code"
                ), relationship = "many-to-many") %>% filter(!is.na(MW))

  final_fossil_costs <- selectable_scenarios %>%
    select(c(Utility_ID_Econ,ba_code,scenario,is_selected)) %>%
    inner_join(final_fossil_costs_by_utility_ba %>%
                 filter(operating_year>cur_build_year), by = c(
                   "Utility_ID_Econ",
                   "ba_code",
                   "scenario"
                 ))

  final_fossil_costs <- rbind(final_fossil_costs,fossil_selected_years)

  rm(fossil_selected_years,
     final_data_inputs_selected_years)

  gc()

  final_data_inputs <- final_data_inputs %>%
    left_join(data_input_parameters, by = c(
      "Utility_ID",
      "Utility_ID_Econ",
      "build_year",
      "Technology_FERC",
      "entity_type",
      "operating_year",
      "energy_community",
      "category",
      "proposed",
      "technology_description",
      "NREL_Class"
    ), relationship = "many-to-many")

  # Allow asset owner with similar wind or solar assets to preferentially curtail the units not receiving PTCs to maximize tax credit uptake.
  # First, we determine for each scenario

  # final_data_inputs <- final_data_inputs %>%
  #   group_by(asset_type_scen_opyear_index) %>% # groups by asset type for a given operational year within each scenario, allowing sums of just potentially selectable assets
  #   mutate(
  #     MWh_no_curt_scen = sum((PTC_Year & !(ITC_Better | is_selected)) * MWh_no_curt, na.rm = TRUE),
  #     MWh_scen = sum((PTC_Year & !(ITC_Better | is_selected)) * MWh, na.rm = TRUE),
  #     MWh_PTC_curt_scen = MWh_no_curt_scen - MWh_scen, # Sums curtailed MWh that would have gotten PTC from all assets of the same type that haven't yet been selected
  #     MWh_no_PTC_scen = sum((PTC_Asset & (ITC_Better | !PTC_Year) & !is_selected) * MWh_no_curt, na.rm = TRUE), # Sums non-PTC MWh from not yet selected assets of the same type
  #   ) %>% ungroup()
  #
  # gc()
  #
  # final_data_inputs <- final_data_inputs %>%
  #   group_by(asset_type_opyear_index) %>% # groups by asset type for a given operational year, but across all scenarios to capture already selected assets from previous build years
  #   mutate(
  #     MWh_PTC_curt = MWh_PTC_curt_scen + sum((PTC_Year & is_selected & !ITC_Better) * (MWh_no_curt - MWh), na.rm = TRUE), # Sums curtailed MWh that would have gotten PTC from all assets of the given type
  #     MWh_no_PTC = MWh_no_PTC_scen + sum((PTC_Asset & is_selected & (ITC_Better | !PTC_Year)) * MWh_no_curt, na.rm = TRUE), # Sums raw non-PTC MWh that could be curtailed from already selected assets of the given type
  #     Frac_MWh_PTC_Assets = if_else(MWh_no_curt_scen>0,(PTC_Year & !(is_selected | ITC_Better)) * MWh_no_curt/MWh_no_curt_scen,0), # Distributes MWh from non-PTC assets to PTC-eligible assets of given type
  #     MWh_PTC = (PTC_Year & !ITC_Better) * (MWh + pmin(Frac_MWh_PTC_Assets * pmin(MWh_PTC_curt,MWh_no_PTC),MWh_no_curt-MWh)) # Reduces curtailment as much as possible for PTC-eligible assets
  #   ) %>% ungroup() # select(-c(MWh_PTC_curt,MWh_no_PTC,Frac_MWh_PTC_Assets))


  final_data_inputs <- final_data_inputs %>%
    mutate(
      ccs_eligible = coalesce(ccs_eligible, FALSE),
      MWh_PTC = MWh,
      owned_redispatch_mmbtu = owned_redispatch_mmbtu * case_when(
        ccs_eligible & category=="patio_clean" ~ CCS_Parasitic_Load, # incremental fuel use with CCS
        category=="proposed_fossil" | category=="proposed_clean" ~ 1,
        TRUE ~ 0
      ),
      owned_redispatch_co2_tonne = owned_redispatch_co2_tonne * case_when(
        ccs_eligible & category=="patio_clean" ~ CCS_Parasitic_Load + (1+CCS_Parasitic_Load) * (-CCS_CO2_Frac), # change in CO2 emissions with CCS
        category=="proposed_fossil" | category=="proposed_clean" ~ 1,
        TRUE ~ 0
      ),
      owned_co2_reduced = case_when(
        ccs_eligible & category=="patio_clean" ~ -owned_redispatch_co2_tonne, # change in CO2 emissions with CCS
        TRUE ~ owned_co2_reduced
      ),
      owned_redispatch_cost_fuel = owned_redispatch_cost_fuel * case_when(
        ccs_eligible & category=="patio_clean" ~ CCS_Parasitic_Load, # incremental fuel costs with CCS
        category=="proposed_fossil" | category=="proposed_clean" ~ 1,
        TRUE ~ 0
      ),
      owned_redispatch_cost_startup = owned_redispatch_cost_startup * case_when(
        ccs_eligible & category=="patio_clean" ~ 0 * CCS_Parasitic_Load, # we assume no incremental startup costs with CCS
        category=="proposed_fossil" | category=="proposed_clean" ~ 1,
        TRUE ~ 0
      ),
      owned_stored_co2_tonne = owned_redispatch_co2_tonne * case_when(
        ccs_eligible & category=="patio_clean" ~ (1+CCS_Parasitic_Load) * CCS_CO2_Frac, # stored CO2 emissions with CCS
        category=="proposed_fossil" | category=="proposed_clean" ~ 1,
        TRUE ~ 0
      )
    )

  final_data_inputs <- final_data_inputs %>%
    mutate(
      MW = if_else((!is.na(NREL_Class)) & (!is.na(CF_improvement)),
                                  MW / CF_improvement,
                                  MW),
      Clean_CapEx = (1 + supply_chain_tariff_CAPEX_impact) * CAPEX * MW * 1000,
      cf_actual = case_when(operating_year %% 4 == 0 & MW>0 ~ (MWh / (MW * 8784)),
                            MW>0 ~ (MWh / (MW * 8760)),
                            TRUE ~ NA)
    ) %>%
    select(-c(base_CF,CF))

  gc()

  final_data_inputs <- final_data_inputs %>%
    mutate(
      Fuel_Costs = (owned_redispatch_cost_fuel + Fuel * cf_actual * (8.76+(operating_year %% 4==0)*0.024) * MW * 1000) * Inflator,
      VOM = `Variable O&M` * cf_actual * (8.76+(operating_year %% 4==0)*0.024) * MW * 1000 * Inflator,
      FOM = `Fixed O&M` * MW * 1000 * Inflator,
      Startup_Costs = owned_redispatch_cost_startup * Inflator,
      Opex = (owned_redispatch_cost_fuel +
                      (`Fixed O&M` +
                         (`Variable O&M` + Fuel) * cf_actual * (8.76+(operating_year %% 4==0)*0.024)) * MW * 1000) * Inflator,
      Clean_PTC = - can_claim_credit * MWh_PTC * PTC_Year *
        (1 + (Domestic_Content_Percent + energy_community) * 0.1) *
        round(2 * PTC * PTC_Inflator_2022 * PTC_Value * Inflation_Factor_2021 / Inflator_2022) *
        0.5 * if_else(transferability_eligible,Transferability_Discount,1),
      Clean_45Q = - (can_claim_credit * ccs_eligible * (cur_project_year <= Duration_45Q)) *
        if_else(transferability_eligible & cur_project_year > 5,Transferability_Discount,1) *
        owned_stored_co2_tonne *
        round(Value_45Q * if_else(operating_year>=Inflation_Start_Year_45Q,Inflation_Factor_2021/Inflator_2025,1))
    )

  final_data_inputs <- final_data_inputs %>%
    group_by(index) %>%
    mutate(
      ITC_Better = (-sum(Clean_PTC * Disc_Fin,na.rm = TRUE) <  ITC_reg * Clean_CapEx) & (ITC_reg>0)
    ) %>% ungroup()

  # write_parquet(final_data_inputs, "final_data_inputs_test_early.parquet")
  # final_data_inputs <- read_parquet("final_data_inputs_test_early.parquet")

  utility_ba_co2_reductions <- owned_co2_reductions %>%
    select(-c(owned_co2_reduced_patio)) %>%
    rename(Utility_ID_Econ = Utility_ID) %>%
    group_by(Utility_ID_Econ,ba_code,scenario,year) %>%
    summarize(co2_red = sum(owned_co2_reduced,na.rm = TRUE)) %>%
    left_join(scenario_selected %>%
                select(c(Utility_ID_Econ,ba_code,build_year,scenario,is_selected)) %>%
                mutate(year=build_year+1) %>%
                select(-c(build_year)), by = c(
      "Utility_ID_Econ",
      "ba_code",
      "year",
      "scenario"
    )) %>% filter(is.na(is_selected) | is_selected) %>%
    group_by(Utility_ID_Econ,ba_code) %>%
    mutate(co2_red_selected = sum(co2_red * is_selected * (year>irp_year & year<=cur_build_year), na.rm = TRUE)) %>% ungroup() %>%
    filter(is.na(is_selected)) %>%
    group_by(Utility_ID_Econ,ba_code,scenario) %>%
    mutate(co2_red = sum(co2_red * (year>irp_year & year<=irp_year+NPV_Duration), na.rm = TRUE) + co2_red_selected) %>% ungroup() %>%
    select(-c(co2_red_selected,is_selected,year)) %>% distinct() %>%
    rename(Utility_ID = Utility_ID_Econ)

  coop_public_capital_limits <- final_data_inputs %>%
    filter(public_utility | coop,
           operating_year==EIR_NewERA_fin_build_year+1,
           build_year<=EIR_NewERA_fin_build_year) %>%
    select(c(
      Utility_ID,
      ba_code,
      scenario,
      coop,
      proposed,
      is_selected,
      Clean_CapEx,
      ITC,
      ITC_Better,
      equity_ratio,
      coop_debt,
      coop_equity
    )) %>%
    group_by(
      Utility_ID,
      ba_code,
      scenario,
      coop,
    ) %>%
    summarize(
      ITC_Required = sum(Clean_CapEx * ITC * ITC_Better, na.rm = TRUE),
      ITC_Available = sum(Clean_CapEx * ITC, na.rm = TRUE) - ITC_Required,
      Clean_CapEx = sum(Clean_CapEx, na.rm = TRUE),
      equity_ratio = mean(equity_ratio, na.rm = TRUE),
      coop_debt = mean(coop_debt, na.rm = TRUE),
      coop_equity = mean(coop_equity, na.rm = TRUE)
    ) %>% ungroup() %>%
    left_join(utility_ba_co2_reductions, by = c(
      "Utility_ID",
      "ba_code",
      "scenario"
    ))

  ## Collapse capital limits to the Utility_ID (not the Utility_ID_Econ, as we are focused on capital limits for
  ## utility-ownership rather than costs for the utility)

  coop_public_capital_limits <- coop_public_capital_limits %>%
    group_by(Utility_ID,scenario,coop) %>%
    summarize(
      ITC_Required = sum(ITC_Required, na.rm = TRUE),
      ITC_Available = sum(ITC_Available, na.rm = TRUE),
      Clean_CapEx = sum(Clean_CapEx, na.rm = TRUE),
      co2_red = sum(co2_red, na.rm = TRUE),
      equity_ratio = mean(equity_ratio, na.rm = TRUE),
      coop_debt = mean(coop_debt, na.rm = TRUE),
      coop_equity = mean(coop_equity, na.rm = TRUE)
    ) %>% ungroup()

  coop_public_capital_limits <- coop_public_capital_limits %>%
    group_by(scenario,coop) %>%
    mutate(
      tot_co2_red = sum(co2_red, na.rm = TRUE),
      NewERA = if_else((co2_red>=100) & coop,
                       pmin(New_ERA_Total * 10^8,
                            New_ERA_Total * (10^9) * co2_red/tot_co2_red,
                            0.25 * coalesce(Clean_CapEx,0)),0),
      NewERA_Deficit = (New_ERA_Total * 10^9 - sum(NewERA, na.rm = TRUE))
    ) %>% ungroup()

  for(i in 1:20){
    coop_public_capital_limits <- coop_public_capital_limits %>%
      group_by(scenario,coop) %>%
      mutate(
        NewERA = if_else((NewERA_Deficit > 10^7) & (co2_red>=100) & coop,
                         pmin(New_ERA_Total * 10^8,
                              (New_ERA_Total * 10^9 + NewERA_Deficit) * co2_red/tot_co2_red,
                              0.25 * coalesce(Clean_CapEx,0)),NewERA),
        NewERA_Deficit = coop * (New_ERA_Total * 10^9 - sum(NewERA, na.rm = TRUE))
      ) %>% ungroup()
  }

  # Here, we incorporate the logic of how a co-op would chose to use a combination of the NewERA grant, loan, and ITC
  # direct pay to meet its total clean capital needs. To do this, we assume that the co-op is going to be looking to
  # use NewERA lending to cover as much of the clean capital costs as possible, and that it would use the grant to
  # cover equity needs, and choose the ITC over the PTC to cover any additional equity needs. We leave the choice of
  # whether to use 0%, 2%, or Treasury NewERA lending as a global input parameter for this analysis. Finally,
  # as a failsafe, we also check if the capital needs are so high that the co-op requires all the NewERA funding
  # as a grant in addition to all the ITC, and allow for full grant instead of debt.

  coop_public_capital_limits <- coop_public_capital_limits %>%
    mutate(
      NewERA_per_tonne = if_else(coop & (co2_red>0), NewERA / co2_red,0),
      NewERA_award = if_else((co2_red>0) & coop,
                                  pmin(New_ERA_Total * 10^8,
                                       NewERA_per_tonne * co2_red,
                                       0.25 * Clean_CapEx),0),
      NewERA_Frac = if_else(coop & (Clean_CapEx>0),pmax(0,((NewERA_award / Clean_CapEx) - NewERA_subsidy_cost)/(1-NewERA_subsidy_cost)),0),
      NewERA_Debt_Frac = if_else(coop & (Clean_CapEx>0),pmin(1,((NewERA_award / Clean_CapEx) - NewERA_Frac) / NewERA_subsidy_cost),0),
      equity_needs = (Min_Pub_Equity_Ratio * Clean_CapEx) -
        if_else(is.na(coop_equity),0,(coop_equity - (coop_debt+coop_equity)*(Min_Pub_Equity_Ratio))) -
        (NewERA_Frac * Clean_CapEx + ITC_Required),
      ITC_Frac_Req = pmax(0,equity_needs/ITC_Available),
      NewERA_Frac = if_else((coop & (Clean_CapEx>0)) & (is.na(ITC_Frac_Req) | ITC_Frac_Req>1),(NewERA_award / Clean_CapEx),NewERA_Frac),
      NewERA_Debt_Frac = if_else(is.na(ITC_Frac_Req) | ITC_Frac_Req>1,0,NewERA_Debt_Frac),
      equity_needs = if_else(is.na(ITC_Frac_Req) | ITC_Frac_Req>1,(Min_Pub_Equity_Ratio * Clean_CapEx) -
        if_else(is.na(coop_equity),0,(coop_equity - (coop_debt+coop_equity)*(Min_Pub_Equity_Ratio))) -
        (NewERA_Frac * Clean_CapEx + ITC_Required),equity_needs),
      ITC_Frac_Req = pmax(0,equity_needs/ITC_Available)
    )

  final_data_inputs <- final_data_inputs %>%
    left_join(coop_public_capital_limits %>% select(c(
      Utility_ID,
      scenario,
      #entity_type,
      ITC_Frac_Req,
      NewERA_Debt_Frac,
      NewERA_Frac
      #equity_needs
    )), by = c(
      "Utility_ID","scenario"
    )) %>%
    mutate(
      Loan_Rates = case_when(
        (is.na(ITC_Frac_Req) | ITC_Frac_Req>1) & (coop | public_utility) ~ BBB_Loan_Rates,
        TRUE ~ A_Loan_Rates
      ),
      equity_ratio = case_when(
        (is.na(ITC_Frac_Req) | ITC_Frac_Req>1) & (coop | public_utility) ~ Min_Pub_Equity_Ratio+0.05,
        TRUE ~ equity_ratio
      ),
      roe = case_when(
        (is.na(ITC_Frac_Req) | ITC_Frac_Req>1) & (coop | public_utility) ~ 0.07,
        TRUE ~ roe
      ),
      ITC_Frac_Req = case_when(
        (!is.na(ITC_Frac_Req)) & (ITC_Frac_Req<=1)  ~ ITC_Frac_Req,
        (is.na(ITC_Frac_Req) | (ITC_Frac_Req>1)) & (coop | public_utility) ~ 1,
        TRUE ~ 0
      ),
      NewERA_Frac = if_else((build_year>=irp_year) &
                              (build_year<=EIR_NewERA_fin_build_year) &
                              !is.na(NewERA_Frac),NewERA_Frac,0),
      NewERA_Debt_Frac = if_else((build_year>=irp_year) &
                              (build_year<=EIR_NewERA_fin_build_year) &
                              !is.na(NewERA_Debt_Frac),NewERA_Debt_Frac,0),
      tax_equity_used = if_else(category %in% c("patio_clean","proposed_clean"),(ITC_Frac_Req>1) | tax_equity_used, FALSE)
    )

  final_data_inputs <- final_data_inputs %>%
    mutate(
      Clean_PTC = if_else(ITC_Better,0,(1-ITC_Frac_Req)) * Clean_PTC,
      ITC_claim = if_else(ITC_Better,1,ITC_Frac_Req) * ITC_claim,
      ITC_reg = if_else(ITC_Better,1,ITC_Frac_Req) * ITC_reg,
      ITC = if_else(ITC_Better,1,ITC_Frac_Req) * ITC,
      equity_ratio = pmax(equity_ratio,ITC+NewERA_Frac),
      debt_cost = case_when(
        public_utility & build_year<=EIR_NewERA_fin_build_year ~ Loan_Rates * (1-Public_EIR_Frac)+ DOE_Loan_Rates * Public_EIR_Frac,
        coop & NewERA_Debt_Frac>0 & build_year<=EIR_NewERA_fin_build_year ~ debt_cost * (1-NewERA_Debt_Frac) + NewERA_Debt_Frac *
                              case_when(
                                NewERA_Rate_Chosen == "Zero" ~ 0,
                                NewERA_Rate_Chosen == "Two" ~ 0.02,
                                TRUE ~ Forward_Treasury_Rates
                              ),
        (debt_spread == "BBB") & !(public_utility | coop) ~ BBB_Loan_Rates,
        (debt_spread == "B") & !(public_utility | coop) ~ B_Loan_Rates,
        (debt_spread == "BB") & !(public_utility | coop) ~ BB_Loan_Rates,
        (debt_spread == "A") & !(public_utility | coop) ~ A_Loan_Rates,
        (debt_spread == "AA") & !(public_utility | coop) ~ AA_Loan_Rates,
        (debt_spread == "AAA") & !(public_utility | coop) ~ AAA_Loan_Rates,
        TRUE ~ debt_cost
      ),
      no_policy_debt_cost = case_when(
        public_utility ~ Loan_Rates,
        (debt_spread == "BBB") & !(public_utility | coop) ~ BBB_Loan_Rates,
        (debt_spread == "B") & !(public_utility | coop) ~ B_Loan_Rates,
        (debt_spread == "BB") & !(public_utility | coop) ~ BB_Loan_Rates,
        (debt_spread == "A") & !(public_utility | coop) ~ A_Loan_Rates,
        (debt_spread == "AA") & !(public_utility | coop) ~ AA_Loan_Rates,
        (debt_spread == "AAA") & !(public_utility | coop) ~ AAA_Loan_Rates,
        TRUE ~ debt_cost
      ),
      adjusted_ror = (roe * equity_ratio) + ((debt_cost * (1 - equity_ratio)) * (1 - Blended_Tax_Rate)),
      no_policy_ror = (roe * equity_ratio) + no_policy_debt_cost * (1 - equity_ratio) * (1 - Blended_Tax_Rate),
      Clean_EIR_OBS_Frac = case_when(Clean_EIR_Off_BS=="Yes" & iou & build_year<=EIR_NewERA_fin_build_year ~ New_Asset_Debt_Replaced_by_EIR+New_Asset_Equity_Replaced_by_EIR,
                                     TRUE ~ 0),
      doe_adjusted_ror = (roe * (equity_ratio - New_Asset_Equity_Replaced_by_EIR * iou * (build_year<=EIR_NewERA_fin_build_year))) +
        (debt_cost * (1 - equity_ratio - New_Asset_Debt_Replaced_by_EIR * iou * (build_year<=EIR_NewERA_fin_build_year)) * (1 - Blended_Tax_Rate)),
      doe_adjusted_ror = case_when(
        Clean_EIR_Off_BS=="Yes" & iou & (build_year<=EIR_NewERA_fin_build_year) ~ doe_adjusted_ror/(1-Clean_EIR_OBS_Frac),
        iou ~ doe_adjusted_ror + (DOE_Loan_Rates * (New_Asset_Equity_Replaced_by_EIR + New_Asset_Debt_Replaced_by_EIR)),
        TRUE ~ doe_adjusted_ror
      ),
      Clean_EIR_PMT = Clean_EIR_OBS_Frac * DOE_Loan_Rates / (1-(1/(1+DOE_Loan_Rates))^Debt_Tenor)
    )

  gc()

  # Step 1c: Pull in MACRS

  final_data_inputs <- final_data_inputs %>%
    mutate(
      cur_MACRS = cur_MACRS * (1 - ITC / 2),
      sum_MACRS = sum_MACRS * (1 - ITC / 2),
      #taxable = (Blended_Tax_Rate > 0),
      #Inflator = Inflator_2021 / Build_Year_Inflator,
      first_year = (cur_project_year == 1),
      utility_debt_year = (cur_project_year <= Debt_Tenor),
      utility_itc_norm_year = (cur_project_year <= ITC_Norm_Period),
      cur_equity_ratio = utility_debt_year * (equity_ratio - New_Asset_Equity_Replaced_by_EIR * iou * (build_year<=EIR_NewERA_fin_build_year)) *
        case_when(
          Clean_EIR_Off_BS=="Yes" & iou & (build_year<=EIR_NewERA_fin_build_year) ~ 1/(1-Clean_EIR_OBS_Frac),
          TRUE ~ 1) + (!utility_debt_year) * equity_ratio,
      cur_ror = utility_debt_year * doe_adjusted_ror + (!utility_debt_year) * adjusted_ror
    ) %>%
    select(-c(`Fixed O&M`, `Variable O&M`, Fuel, Normalized))

  # Step 3a: Renewable cost calculations

  final_data_inputs <- final_data_inputs %>%
    mutate(
      # Term corresponding to clean energy capital costs adjusted to account for the regulatory asset associated with
      # deductibility of 50% of ITC and its amortization as well as repayment of EIR debt through dedicated surcharge

      Clean_Cap_Costs = Clean_CapEx * (
        (((cur_ror / ( 1 - Blended_Tax_Rate)) *
            (1 - cur_project_year / Asset_Life)) +
           1 / Asset_Life) * (1 - Clean_EIR_OBS_Frac) +
          Clean_EIR_PMT),

      # ADIT associated with MACRS, corrected for the regulatory liability associated with deductibility of half of ITC

      Clean_ADIT = - Clean_CapEx *
        ((cur_ror / (1 - Blended_Tax_Rate)) *
           (sum_MACRS - cur_project_year * (1-ITC/2) / Asset_Life)
         * Blended_Tax_Rate),

      # Term accounting for ITC and NewERA, including impact of normalization rules for IOUs, assuming election not to
      # deduct ITC from ratebase and instead pass it through to customers over the life of the asset - unless normalization
      # is repealed. In that case, the ITC must be deducted from ratebase until it is fully passed through to customers
      # over some chosen ITC_Norm_Period. In the case of a coop, this also includes the impact of the NewERA grant. Further,
      # for both co-ops and public utilities, we also attempt to incorporate the assumption that debt coverage covenants
      # may require rates to be higher than required to cover costs alone. However, we then assume that not-for-profit
      # requirements would likely see those overcollections returned within a year either as customer rebates or as return
      # of patronage capital to members. We model this by applying a one year discount at ror to the benefit from ITC / NewERA

      Clean_ITC_NewERA = - Clean_CapEx  *
        (ITC_reg / (1 - Blended_Tax_Rate) + NewERA_Frac) * utility_itc_norm_year *
            (1 / ITC_Norm_Period + cur_ror * (!ITC_Normalized) * (1 - cur_project_year / ITC_Norm_Period)) /
            (1+ror * (public_utility | coop)),

      Clean_Rate_Base = Clean_CapEx * (
        (1 - cur_project_year/Asset_Life) * (1-Clean_EIR_OBS_Frac) - # Clean capital costs
          (sum_MACRS - cur_project_year *(1-ITC/2) / Asset_Life) * Blended_Tax_Rate - #ADIT
          (ITC_reg * (!ITC_Normalized) + NewERA_Frac) * utility_itc_norm_year * # Reducing rate base by ITC over asset life if not an IOU, and over voluntary normalization period if an IOU and ITC tax normalization rules do not apply
              (1 - cur_project_year / ITC_Norm_Period)
      ),

      Clean_Pre_Tax_Earnings = Clean_Rate_Base * cur_equity_ratio * (roe / (1-Blended_Tax_Rate)),

      No_Policy_Capex_Costs = Clean_CapEx * (
        (((no_policy_ror / ( 1 - Blended_Tax_Rate)) *
            (1 - cur_project_year / Asset_Life)) +
           1 / Asset_Life) -
          ((no_policy_ror / (1 - Blended_Tax_Rate)) *
             (sum_MACRS - cur_project_year / Asset_Life)
           * Blended_Tax_Rate))


    )

  gc()

  # Calculation of overall ratebase impact of clean asset, including the potential impact of carryforward
  # of tax benefits using a gross-up method to account for earnings and tax impacts of carryforward tax asset

  final_data_inputs <- final_data_inputs %>%
    group_by(index) %>%
    arrange(cur_project_year) %>%
    mutate(
      Sum_Tax_Liabilities = cumsum(Clean_Pre_Tax_Earnings) * Blended_Tax_Rate
    )

  final_data_inputs <- final_data_inputs %>%
    group_by(index) %>%
    mutate(
      Sum_Tax_Benefits = sum_MACRS * Blended_Tax_Rate * Clean_CapEx +
        (!(public_utility | transferability_eligible | coop)) *
        (Clean_CapEx * first_year * ITC_claim -
           (Clean_PTC * if_else(cur_project_year<=PTC_Duration,cur_project_year,PTC_Duration) +
           Clean_45Q * if_else(cur_project_year<=Duration_45Q,cur_project_year,Duration_45Q))) ,
      Clean_Net_Tax_Asset = if_else(Sum_Tax_Benefits > Sum_Tax_Liabilities,
                                    (Sum_Tax_Benefits - Sum_Tax_Liabilities)/
                                      (1+ cur_equity_ratio * roe * Blended_Tax_Rate / (1-Blended_Tax_Rate)),0),
      Clean_Rate_Base = Clean_Rate_Base + Clean_Net_Tax_Asset,
      Earnings = Clean_Rate_Base * cur_equity_ratio * roe,
      Clean_Tax_Asset = Clean_Net_Tax_Asset * (cur_ror / (1-Blended_Tax_Rate)),
      Capex_Costs = (Clean_PTC + Clean_45Q)  / (1 - Blended_Tax_Rate) + Clean_Cap_Costs + Clean_ADIT + Clean_ITC_NewERA + Clean_Tax_Asset,
      Costs = Opex + Capex_Costs
    ) %>% ungroup()

  gc()

  final_data_inputs <- final_data_inputs %>%
    group_by(index) %>%
    arrange(cur_project_year) %>%
    mutate(
      Forward_Earnings = lead(Earnings, n = 3)
    ) %>% ungroup() %>%
    rename(Emissions = owned_redispatch_co2_tonne,Emissions_Reduced = owned_co2_reduced)

  capital_recycled_by_scenario <- final_data_inputs %>%
    filter(operating_year<=cur_build_year+1,!proposed) %>%
    group_by(Utility_ID,plant_id_eia,generator_id,scenario,operating_year,re_limits_dispatch,re_energy,nuclear_scen,storage_li_pct,ccs_scen) %>%
    summarize(Capital_Recycled = sum(Clean_CapEx * (1-ITC),na.rm = TRUE)) %>% ungroup() %>%
    mutate(build_year=operating_year-1) %>% select(-c(operating_year))

  refi_parquet <- refinancing_parquet %>%
    inner_join(capital_recycled_by_scenario, by = c("Utility_ID","plant_id_eia","generator_id","build_year"), relationship = "many-to-many") %>%
    filter(Capital_Recycled >= Securitization_Amount * refinancing_capital_recycling_trigger) %>%
    group_by(Utility_ID,plant_id_eia,generator_id,scenario) %>%
    filter(build_year == min(build_year,na.rm = TRUE)) %>%
    mutate(
      Utility_ID_Econ=Utility_ID,
      MWh = 0,
      MW = 0,
      re_plant_id = NA,
      ccs_eligible = NA,
      Emissions = 0,
      Emissions_Reduced = 0,
      category = as.factor("refinancing")
    ) %>% ungroup()

  patio_econ <- final_data_inputs %>%
    mutate(MW = (technology_description!="Transmission") * MW) %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      technology_description,
      Technology_FERC,
      operating_year,
      scenario,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    summarize(
      across(c(Costs,Capex_Costs,Opex,MW,MWh,Emissions,Emissions_Reduced), ~ sum(.x, na.rm = TRUE)),
      across(c(Earnings,Forward_Earnings), ~ sum(.x * (Utility_ID_Econ==Utility_ID), na.rm = TRUE))
    ) %>% ungroup()

  refi_econ <- refi_parquet %>%
    mutate(MW = (technology_description!="Transmission") * MW) %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      technology_description,
      Technology_FERC,
      operating_year,
      scenario,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    summarize(
      across(c(Costs,Capex_Costs,Opex,Earnings,Forward_Earnings,MW,MWh,Emissions,Emissions_Reduced), ~ sum(.x, na.rm = TRUE))
    ) %>% ungroup()

  fossil_econ <- final_fossil_costs %>%
    mutate(MW = (technology_description!="Transmission") * MW) %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      operating_year,
      scenario,
      technology_description,
      Technology_FERC,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    summarize(
      across(c(Costs,Capex_Costs,Opex,Earnings,Forward_Earnings,MW,MWh,Emissions,Emissions_Reduced), ~ sum(.x, na.rm = TRUE))
    ) %>% ungroup()

  if(cur_build_year==min(loop_years)) {

    final_counterfactual_outputs <- final_data_inputs %>%
      filter(scenario=="counterfactual") %>%
      mutate(counterfactual_baseline = TRUE) %>%
      left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
      mutate(
        across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,MWh,No_Policy_Capex_Costs),
               ~ .x * Disc, .names = "{.col}_Disc" ),
        across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
        is_irp_year = (operating_year %in% irp_years),
        historical_actuals = FALSE
      )

    counterfactual_econ_by_tech <- rbind(fossil_econ %>% filter(scenario=="counterfactual"),
                                 patio_econ  %>% filter(scenario=="counterfactual") ) %>%
      group_by(
        Utility_ID_Econ,
        ba_code,
        category,
        technology_description,
        Technology_FERC,
        operating_year,
        scenario,
        re_energy,
        nuclear_scen,
        storage_li_pct,
        ccs_scen,
        re_limits_dispatch
      ) %>%
      summarize(
        across(c(Costs,Capex_Costs,Opex,Earnings,Forward_Earnings,MW,MWh,Emissions,Emissions_Reduced), ~ sum(.x, na.rm = TRUE))
      ) %>% ungroup() %>%
      left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
      filter(operating_year %in% irp_years) %>%
      group_by(
        Utility_ID_Econ,
        ba_code,
        category,
        scenario,
        technology_description,
        Technology_FERC,
        re_energy,
        nuclear_scen,
        storage_li_pct,
        ccs_scen,
        re_limits_dispatch
      ) %>%
      mutate(
        across(c(Costs,Capex_Costs,Opex,MWh),
               ~ .x * Disc, .names = "{.col}_Disc" ),
        across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
        across(c(Costs,Capex_Costs,Opex,MWh),
               ~ sum(.x * Disc, na.rm = TRUE), .names = "{.col}_Disc_Sum" ),
        across(c(Earnings), ~ sum(.x * Disc_E, na.rm = TRUE), .names = "{.col}_Disc_Sum" )
      ) %>% ungroup()

    counterfactual_econ <- counterfactual_econ_by_tech %>%
      group_by(
        Utility_ID_Econ,
        ba_code,
        scenario,
        re_energy,
        nuclear_scen,
        storage_li_pct,
        ccs_scen,
        re_limits_dispatch
      ) %>%
      summarize(
        across(c(Costs_Disc,Capex_Costs_Disc,Opex_Disc,MWh_Disc,Earnings_Disc,Emissions,Emissions_Reduced),
               ~ sum(.x, na.rm = TRUE), .names = "{.col}_Sum" ),
        across(c(Forward_Earnings,Costs,Capex_Costs,Opex),
               ~ sum(.x * (operating_year==irp_year+1), na.rm = TRUE), .names = "{.col}_irp_year"),
        across(c(Emissions,Emissions_Reduced),
               ~ sum(.x * (operating_year==target_build_year+1), na.rm = TRUE), .names = "{.col}_fin_build_year"),
        MW = max(MW, na.rm = TRUE)
      ) %>% ungroup() %>%
      mutate(
        Levelized_Costs = round(if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,Costs_Disc_Sum), digits = 1),
        counterfactual_baseline = TRUE,
        patio_earnings = FALSE,
        net_patio_earnings = FALSE
      )

  }

  final_econ_by_tech <- rbind(patio_econ,refi_econ,fossil_econ) %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      technology_description,
      Technology_FERC,
      operating_year,
      scenario,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    summarize(
      across(c(Costs,Capex_Costs,Opex,Earnings,Forward_Earnings,MW,MWh,Emissions,Emissions_Reduced), ~ sum(.x, na.rm = TRUE))
    ) %>% ungroup() %>%
    mutate(scenario = as.factor(scenario)) %>%
    left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
    filter(operating_year %in% irp_years) %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      scenario,
      technology_description,
      Technology_FERC,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    mutate(
      across(c(Costs,Capex_Costs,Opex,MWh),
             ~ .x * Disc, .names = "{.col}_Disc" ),
      across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
      across(c(Costs,Capex_Costs,Opex,MWh),
             ~ sum(.x * Disc, na.rm = TRUE), .names = "{.col}_Disc_Sum" ),
      across(c(Earnings), ~ sum(.x * Disc_E, na.rm = TRUE), .names = "{.col}_Disc_Sum" )
    ) %>% ungroup()

  final_econ_by_category <- final_econ_by_tech %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      category,
      scenario,
    ) %>%
    summarize(
      across(c(Costs_Disc,Capex_Costs_Disc,Opex_Disc,MWh_Disc,Earnings_Disc,Emissions,Emissions_Reduced),
             ~ sum(.x, na.rm = TRUE), .names = "{.col}_Sum" ),
      across(c(Forward_Earnings,Costs,Capex_Costs,Opex),
             ~ sum(.x * (operating_year==irp_year+1), na.rm = TRUE), .names = "{.col}_irp_year"),
      across(c(Emissions,Emissions_Reduced),
             ~ sum(.x * (operating_year==target_build_year+1), na.rm = TRUE), .names = "{.col}_fin_build_year"),
      MW = max(MW, na.rm = TRUE)
    ) %>% ungroup() %>%
    mutate(
      Levelized_Costs = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,Costs_Disc_Sum),
      counterfactual_baseline = FALSE
    )

  final_econ <- final_econ_by_tech %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      scenario,
      re_energy,
      nuclear_scen,
      storage_li_pct,
      ccs_scen,
      re_limits_dispatch
    ) %>%
    summarize(
      patio_earnings = (sum(Earnings_Disc * (category %in% c("patio_clean","transmission_lines","patio_transmission_upgrades")), na.rm = TRUE)>0),
      net_patio_earnings = (sum(Earnings_Disc * (category %in% c("patio_clean","transmission_lines","refinancing","patio_transmission_upgrades")), na.rm = TRUE)>0),
      across(c(Costs_Disc,Capex_Costs_Disc,Opex_Disc,MWh_Disc,Earnings_Disc,Emissions,Emissions_Reduced),
             ~ sum(.x, na.rm = TRUE), .names = "{.col}_Sum" ),
      across(c(Forward_Earnings,Costs,Capex_Costs,Opex),
             ~ sum(.x * (operating_year==irp_year+1), na.rm = TRUE), .names = "{.col}_irp_year"),
      across(c(Emissions,Emissions_Reduced),
             ~ sum(.x * (operating_year==target_build_year+1), na.rm = TRUE), .names = "{.col}_fin_build_year"),
      MW = max(MW, na.rm = TRUE)
    ) %>% ungroup() %>%
    mutate(
      Levelized_Costs = round(if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,0),digits = 1),
      counterfactual_baseline = FALSE
    )

  final_econ <- rbind(final_econ,counterfactual_econ) %>%
    left_join(cur_scenario_selected %>% select(Utility_ID_Econ,ba_code,scenario,prev_is_selected = is_selected),
              by = c("Utility_ID_Econ","ba_code","scenario")) %>%
    group_by(
      Utility_ID_Econ,
      ba_code
    ) %>%
    mutate(
      prev_is_selected = coalesce(prev_is_selected & !counterfactual_baseline,FALSE),
      Cost_Reduction = sum(Levelized_Costs * counterfactual_baseline, na.rm = TRUE) - Levelized_Costs,
      Savings_Fraction = Cost_Reduction/sum(Levelized_Costs * counterfactual_baseline, na.rm = TRUE),
      Min_Costs = ((min(if_else(counterfactual_baseline,NA,Levelized_Costs), na.rm = TRUE)) == Levelized_Costs),
      Max_Earnings = (Earnings_Disc_Sum==max(Earnings_Disc_Sum *
                                               ((Savings_Fraction>=(utility_savings_overshoot-1)) | Min_Costs) *
                                               (!counterfactual_baseline), na.rm = TRUE )),
      Max_Cost_Reduction = (Cost_Reduction==max(Cost_Reduction * Max_Earnings * (!counterfactual_baseline), na.rm = TRUE)),
      Max_Earnings_Ties = (sum(Max_Earnings & !counterfactual_baseline, na.rm = TRUE)>1),
      Max_Cost_Reduction_Ties = (sum(Max_Earnings_Ties & Max_Cost_Reduction & !counterfactual_baseline, na.rm = TRUE)>1)
    ) %>% ungroup() %>%
    group_by(
      Utility_ID_Econ,
      ba_code,
      Max_Earnings,
      Max_Cost_Reduction
    ) %>%
    mutate(
      Min_Feasible = Max_Earnings & Max_Cost_Reduction &
        (re_energy==min(re_energy, na.rm = TRUE)) &
        (storage_li_pct==min(storage_li_pct, na.rm = TRUE)) &
        (nuclear_scen==min(nuclear_scen, na.rm = TRUE)) &
        (ccs_scen==min(ccs_scen, na.rm = TRUE)) & !counterfactual_baseline,
      is_selected = Max_Earnings &
        ((!Max_Earnings_Ties) | (Max_Cost_Reduction & Max_Earnings_Ties)) &
        ((!Max_Cost_Reduction_Ties) | (Min_Feasible & Max_Cost_Reduction_Ties)) &
        !counterfactual_baseline,
      is_selected = coalesce(is_selected,FALSE)
    ) %>% ungroup() %>%
    group_by(
      Utility_ID_Econ,
      ba_code
    ) %>%
    mutate(
      tot_sel = sum(is_selected, na.rm = TRUE),
      tot_prev_sel = sum(prev_is_selected, na.rm = TRUE),
      is_selected = case_when(
        tot_sel==0 & tot_prev_sel==0 ~ (scenario=="counterfactual") & !counterfactual_baseline,
        tot_sel==0 & tot_prev_sel>0 ~ prev_is_selected,
        TRUE ~ is_selected
      ),
      tot_sel = sum(is_selected, na.rm = TRUE),
      build_year = cur_build_year,
      re_energy_max = max(re_energy * is_selected),
      storage_li_pct_max = max(storage_li_pct * is_selected),
      re_still_limits_dispatch = as.logical(min(if_else(is_selected,re_limits_dispatch,TRUE), na.rm = TRUE)),
      nuclear_scen_max = max(nuclear_scen * is_selected),
      ccs_scen_max = max(ccs_scen * is_selected)
    ) %>% ungroup()

  final_econ_ba <- final_econ %>%
    group_by(ba_code,scenario,counterfactual_baseline) %>%
    summarize(
      across(c(Capex_Costs_Disc_Sum,Opex_Disc_Sum,Costs_Disc_Sum,MWh_Disc_Sum,Earnings_Disc_Sum),~sum(.x, na.rm = TRUE), .names = "ba_{.col}" ),
      ba_Earnings_Challenge = coalesce((
        sum(Costs_Disc_Sum * (patio_earnings & net_patio_earnings), na.rm = TRUE) <
          ba_earnings_threshold * sum(Costs_Disc_Sum * patio_earnings, na.rm = TRUE)),FALSE)
    ) %>% ungroup() %>%
    group_by(ba_code) %>%
    mutate(
      across(c(ba_Costs_Disc_Sum,ba_MWh_Disc_Sum,ba_Earnings_Disc_Sum),~sum(.x * counterfactual_baseline, na.rm = TRUE), .names = "{.col}_count" ),
      ba_Levelized_Costs_count = if_else(pmin(ba_MWh_Disc_Sum_count,ba_MWh_Disc_Sum)>0,ba_Costs_Disc_Sum_count/pmin(ba_MWh_Disc_Sum_count,ba_MWh_Disc_Sum),NA),
      ba_Levelized_Costs = if_else(pmin(ba_MWh_Disc_Sum_count,ba_MWh_Disc_Sum)>0,ba_Costs_Disc_Sum/pmin(ba_MWh_Disc_Sum_count,ba_MWh_Disc_Sum),NA),
      ba_Cost_Reduction =  ba_Levelized_Costs_count - ba_Levelized_Costs,
      ba_Min_Costs = coalesce((scenario[which.max(if_else(counterfactual_baseline,-Inf,ba_Cost_Reduction))] == scenario) & !counterfactual_baseline,FALSE),
      ba_Max_Earnings = coalesce((scenario[which.max(if_else(counterfactual_baseline | ba_Earnings_Challenge |
                                                               (coalesce(ba_Cost_Reduction<ba_savings_tolerance,FALSE) & !ba_Min_Costs),-Inf,ba_Earnings_Disc_Sum))] == scenario) & !counterfactual_baseline,FALSE),
      ba_is_selected = ((ba_least_cost & ba_Min_Costs) | (ba_Max_Earnings & !ba_least_cost)) & !counterfactual_baseline,
      ba_is_selected = coalesce(ba_is_selected,FALSE)
    ) %>% ungroup()

  final_econ <- final_econ %>%
    left_join(final_econ_ba, by = c("ba_code","scenario","counterfactual_baseline")) %>%
    group_by(Utility_ID_Econ,ba_code) %>%
    mutate(
      ba_still_counterfactual = sum(ba_is_selected * (scenario=="counterfactual"), na.rm = TRUE)>0,
      utility_wants_to_build = sum(is_selected * (scenario!="counterfactual"), na.rm = TRUE)>0,
      ba_tot_sel = sum(ba_is_selected, na.rm = TRUE),
      ba_is_selected = case_when(
        utility_override_for_counterfactual_ba & ba_tot_sel==0 & tot_sel==0 & tot_prev_sel==0 ~ (scenario=="counterfactual") & !counterfactual_baseline,
        utility_override_for_counterfactual_ba & ba_tot_sel==0 & tot_sel==0 & tot_prev_sel>0 ~ prev_is_selected,
        utility_override_for_counterfactual_ba & ba_tot_sel==0 & tot_sel>0 ~ is_selected,
        utility_override_for_counterfactual_ba & ba_tot_sel>0 & ba_still_counterfactual & utility_wants_to_build ~ is_selected,
        ba_tot_sel==0 & tot_prev_sel>0 ~ prev_is_selected,
        ba_tot_sel==0 & tot_prev_sel==0 ~ (scenario=="counterfactual") & !counterfactual_baseline,
        TRUE ~ ba_is_selected
      ),
      ba_tot_sel = sum(ba_is_selected, na.rm = TRUE),
      ba_re_energy_max = max(re_energy * ba_is_selected),
      ba_storage_li_pct_max = max(storage_li_pct * ba_is_selected),
      ba_re_still_limits_dispatch = as.logical(min(if_else(ba_is_selected,re_limits_dispatch,TRUE), na.rm = TRUE)),
      ba_nuclear_scen_max = max(nuclear_scen * ba_is_selected),
      ba_ccs_scen_max = max(ccs_scen * ba_is_selected)
    ) %>% ungroup()

  if(ba_scenario_selection) {
    cur_scenario_selected <- selectable_scenarios %>%
      select(-c(re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max,is_selected)) %>%
      inner_join(final_econ %>% filter(ba_is_selected) %>%
                   filter(!counterfactual_baseline) %>%
                   select(Utility_ID_Econ,ba_code,scenario,is_selected = ba_is_selected,
                          re_energy_max = ba_re_energy_max,
                          storage_li_pct_max = ba_storage_li_pct_max,
                          re_still_limits_dispatch = ba_re_still_limits_dispatch,
                          nuclear_scen_max = ba_nuclear_scen_max,
                          ccs_scen_max = ba_ccs_scen_max),
                 by = c("Utility_ID_Econ","ba_code","scenario")) %>%
      mutate(scenario = as.factor(scenario))

  } else {
    cur_scenario_selected <- selectable_scenarios %>%
    select(-c(re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max,is_selected)) %>%
    inner_join(final_econ %>% filter(is_selected) %>%
                 filter(!counterfactual_baseline) %>%
                 select(Utility_ID_Econ,ba_code,scenario,is_selected,
                        re_energy_max,storage_li_pct_max,re_still_limits_dispatch,nuclear_scen_max,ccs_scen_max),
               by = c("Utility_ID_Econ","ba_code","scenario")) %>%
    mutate(scenario = as.factor(scenario))
  }

  scenario_selected <- rbind(scenario_selected,cur_scenario_selected)

  paste("Processed build_year = ",cur_build_year,sep="")
  gc()

}

additional_years <- c((max(loop_years)+2):(irp_year+NPV_Duration))

scenarios_selected_by_year <- rbind(scenario_selected %>%
  select(Utility_ID_Econ,ba_code,scenario,operating_year = build_year,is_selected) %>%
    distinct() %>% mutate(operating_year = operating_year+1),
  cur_scenario_selected %>%  select(Utility_ID_Econ,ba_code,scenario,is_selected) %>% distinct() %>%
    cross_join(as.data.frame(additional_years), copy = TRUE) %>%
    rename(operating_year = additional_years)) %>% mutate(scenario = as.factor(scenario))

final_econ_by_tech <- rbind(
  final_econ_by_tech %>% mutate(counterfactual_baseline = FALSE),
  counterfactual_econ_by_tech %>% mutate(counterfactual_baseline = TRUE)) %>%
  mutate(scenario = as.factor(scenario)) %>%
  left_join(scenarios_selected_by_year, by = c(
    "Utility_ID_Econ",
    "ba_code",
    "scenario",
    "operating_year"
  ))

final_data_outputs <- final_data_inputs %>%
  select(-c(is_selected)) %>%
  left_join(cur_scenario_selected %>% select(c(Utility_ID_Econ,ba_code,scenario,is_selected)), by = c(
    "Utility_ID_Econ",
    "ba_code",
    "scenario"
  )) %>%
  left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
  mutate(
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,MWh,No_Policy_Capex_Costs),
           ~ .x * Disc, .names = "{.col}_Disc" ),
    across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
    is_irp_year = (operating_year %in% irp_years)
  )


gc()
gc()

final_prop_and_clean_outputs <- final_data_outputs %>%
  filter(is_selected,!is.na(is_selected)) %>%
  mutate(
    counterfactual_baseline = FALSE,
    historical_actuals = FALSE,
  ) %>% rbind(final_counterfactual_outputs) %>%
  mutate(
    Securitization_Amount = NA,
    Securitization_Rate = NA,
    Securitization_PMT = NA,
    Securitization_OC = NA
  ) %>% rename(Capex = Clean_CapEx)

if(ba_scenario_selection) {
  scenarios_selected_by_year_ba <- scenarios_selected_by_year %>%
    select(-c(Utility_ID_Econ)) %>% distinct()

  final_fossil_outputs <- fossil_parquet %>%
    left_join(scenarios_selected_by_year_ba, by = c("ba_code","scenario","operating_year")) %>%
    filter(is_selected == TRUE) %>%
    mutate(
      counterfactual_baseline = FALSE,
      historical_actuals = FALSE,
      re_generator_id = NA,
      re_plant_id = NA
    )
} else {
  scenarios_selected_by_year_ba <- scenarios_selected_by_year %>%
    add_count(ba_code,operating_year,scenario, name = "nsel") %>%
    select(-c(Utility_ID_Econ)) %>% distinct() %>%
    group_by(ba_code,operating_year) %>%
    mutate(
      is_max = (nsel==max(nsel))
    ) %>% ungroup() %>% filter(is_max) %>% distinct() %>%
    add_count(ba_code,operating_year,nsel, name = "nscens") %>%
    group_by(ba_code,operating_year) %>%
    mutate(is_selected = if_else(nscens==1,TRUE,(scenario==first(scenario)))) %>%
    ungroup() %>% select(-c(nsel,is_max,nscens)) %>%
    filter(is_selected)

  final_fossil_outputs <- fossil_parquet %>%
    left_join(scenarios_selected_by_year, by = c("Utility_ID" = "Utility_ID_Econ","ba_code","scenario","operating_year")) %>%
    left_join(scenarios_selected_by_year_ba %>% rename(is_selected_ba = is_selected), by = c("ba_code","scenario","operating_year")) %>%
    mutate(is_selected = coalesce(is_selected,is_selected_ba)) %>% select(-c(is_selected_ba)) %>%
    filter(is_selected == TRUE) %>%
    mutate(
      counterfactual_baseline = FALSE,
      historical_actuals = FALSE,
      re_generator_id = NA,
      re_plant_id = NA
    )
}

final_fossil_outputs <- final_fossil_outputs %>%
  rbind(fossil_parquet %>%
          filter(scenario %in% c("counterfactual","historical")) %>%
          mutate(
            counterfactual_baseline = (scenario=="counterfactual"),
            historical_actuals = (scenario=="historical"),
            is_selected = FALSE,
            re_generator_id = NA,
            re_plant_id = NA
          )
        ) %>%
  mutate(Utility_ID_Econ = Utility_ID) %>%
  left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
  mutate(
    NREL_Class = NA,
      re_generator_id = NA,
    energy_community = NA,
    Clean_EIR_OBS_Frac = NA,
    Clean_EIR_PMT = NA,
    ITC = NA,
    ITC_claim = NA,
    ITC_reg = NA,
    ITC_Better = NA,
    NewERA_Frac = NA,
    NewERA_Debt_Frac = NA,
    ITC_Frac_Req = NA,
    coop_debt = NA,
    coop_equity = NA,
    doe_adjusted_ror = NA,
    Securitization_Amount = NA,
    Securitization_Rate = NA,
    Securitization_PMT = NA,
    Securitization_OC = NA,
    No_Policy_Capex_Costs = NA,
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,MWh,No_Policy_Capex_Costs),
           ~ .x * Disc, .names = "{.col}_Disc" ),
    across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
    is_irp_year = (operating_year %in% irp_years),
    MWh_no_curt = MWh
  )

final_refinancing_outputs <- refi_parquet %>%
  left_join(cur_scenario_selected %>% select(c(Utility_ID_Econ,ba_code,scenario,final_selected = is_selected,
                                               re_limits_dispatch,re_energy,nuclear_scen,storage_li_pct,ccs_scen)), by = c(
    "Utility_ID_Econ",
    "ba_code",
    "scenario"
  )) %>%
  mutate(
    is_selected = final_selected
  ) %>% select(-c(final_selected)) %>%
  filter(is_selected == TRUE) %>%
  mutate(
    counterfactual_baseline = FALSE,
    historical_actuals = FALSE
  ) %>%
  left_join(Disc_Factors, by = c("Utility_ID_Econ" = "Utility_ID","operating_year")) %>%
  mutate(
    re_generator_id = NA,
    Fuel_Costs = NA,
    FOM = NA,
    VOM = NA,
    Capex = NA,
    Startup_Costs = NA,
    NREL_Class = NA,
    energy_community = NA,
    Clean_EIR_OBS_Frac = NA,
    Clean_EIR_PMT = NA,
    ITC = NA,
    ITC_claim = NA,
    ITC_reg = NA,
    ITC_Better = NA,
    NewERA_Frac = NA,
    NewERA_Debt_Frac = NA,
    ITC_Frac_Req = NA,
    coop_debt = NA,
    coop_equity = NA,
    equity_ratio = NA,
    debt_cost = NA,
    roe = NA,
    adjusted_ror = NA,
    doe_adjusted_ror = NA,
    No_Policy_Capex_Costs = NA,
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,MWh,No_Policy_Capex_Costs),
           ~ .x * Disc, .names = "{.col}_Disc" ),
    across(c(Earnings), ~ .x * Disc_E, .names = "{.col}_Disc" ),
    is_irp_year = (operating_year %in% irp_years),
    MWh_no_curt = MWh
  )

final_outputs_columns <- c(
  "Utility_ID",
  "Utility_ID_Econ",
  "entity_type",
  "ba_code",
  "build_year",
  "operating_year",
  "is_irp_year",
  "counterfactual_baseline",
  "historical_actuals",
  "category",
  "Technology_FERC",
  "technology_description",
  "NREL_Class",
  "re_plant_id",
  "re_generator_id",
  "plant_id_eia",
  "generator_id",
  "scenario",
  #"re_limits_dispatch",
  "energy_community",
  #"re_energy",
  #"nuclear_scen",
  #"storage_li_pct",
  #"ccs_scen",
  "ccs_eligible",
  "ITC_Better",
  "Clean_EIR_OBS_Frac",
  "Clean_EIR_PMT",
  "ITC",
  "ITC_claim",
  "ITC_reg",
  "coop_debt",
  "coop_equity",
  "equity_ratio",
  "debt_cost",
  "roe",
  "adjusted_ror",
  "doe_adjusted_ror",
  "NewERA_Frac",
  "NewERA_Debt_Frac",
  "ITC_Frac_Req",
  "MW",
  "Capex",
  "Costs",
  "Capex_Costs",
  "No_Policy_Capex_Costs",
  "Fuel_Costs",
  "FOM",
  "VOM",
  "Startup_Costs",
  "Opex",
  "MWh",
  "MWh_no_curt",
  "Earnings",
  "Emissions",
  "Emissions_Reduced",
  "Disc",
  "Disc_E",
  "Costs_Disc",
  "Capex_Costs_Disc",
  "No_Policy_Capex_Costs_Disc",
  "Opex_Disc",
  "MWh_Disc",
  "Earnings_Disc",
  "Securitization_Amount",
  "Securitization_Rate",
  "Securitization_PMT",
  "Securitization_OC"
)

final_outputs <- rbind(
  final_prop_and_clean_outputs %>% select(all_of(final_outputs_columns)),
  final_fossil_outputs %>% select(all_of(final_outputs_columns)),
  final_refinancing_outputs %>% select(all_of(final_outputs_columns))
) %>% mutate(
  technology_description = as.factor(technology_description),
  generator_id = as.factor(generator_id),
  scenario = as.factor(scenario),
  refi_trigger = refinancing_capital_recycling_trigger,
  savings_tol = ba_savings_tolerance,
  earnings_thresh = ba_earnings_threshold,
  least_cost = ba_least_cost,
  Debt_Repl_EIR = New_Asset_Debt_Replaced_by_EIR,
  Equity_Repl_EIR = New_Asset_Equity_Replaced_by_EIR
)

scenario_selected <- scenario_selected %>%
  mutate(
    refi_trigger = refinancing_capital_recycling_trigger,
    savings_tol = ba_savings_tolerance,
    earnings_thresh = ba_earnings_threshold,
    least_cost = ba_least_cost,
    Debt_Repl_EIR = New_Asset_Debt_Replaced_by_EIR,
    Equity_Repl_EIR = New_Asset_Equity_Replaced_by_EIR
  )

if(file.exists("final_outputs.parquet")){
  final_outputs_agg <- read_parquet("final_outputs.parquet")
  final_outputs_agg <- rbind(final_outputs_agg,final_outputs %>% filter(!historical_actuals))
  write_parquet(final_outputs_agg,"final_outputs.parquet")
  scenario_selected_agg <- read_parquet("scenario_selected.parquet")
  scenario_selected_agg <- rbind(scenario_selected_agg,scenario_selected)
  write_parquet(scenario_selected_agg,"scenario_selected.parquet")
} else {
  write_parquet(final_outputs,"final_outputs.parquet")
  write_parquet(scenario_selected,"scenario_selected.parquet")
  final_outputs_agg <- read_parquet("final_outputs.parquet")
  scenario_selected_agg <- read_parquet("scenario_selected.parquet")
}

# final_outputs_name <- paste("final_outputs-",patio_results,ifelse(ba_least_cost,"-LC",""), ".parquet",sep="")
# scenario_selected_name <- paste("scenario_selected-",patio_results,".parquet",sep="")
# write_parquet(final_outputs,"final_outputs.parquet")
# write_parquet(scenario_selected,"scenario_selected.parquet")
#
# # file.remove("fossil.parquet")
# # file.remove("refinancing.parquet")
#
# parquet_name_detail <- paste("final_data_outputs-",patio_results,".parquet",sep="")
# write_parquet(final_data_outputs,parquet_name_detail)

# final_data_inputs_sums <- final_data_inputs %>%
#   filter((operating_year==2024)) %>%
#   group_by(scenario,category) %>%
#   summarize(
#     MW = sum(MW,na.rm = TRUE),
#     MW_Selected = sum(MW * is_selected,na.rm = TRUE),
#   )
#
# final_data_outputs_sums <- final_data_outputs %>%
#   filter((operating_year==2032) & category=="patio_clean") %>%
#   group_by(scenario,category) %>%
#   summarize(
#     MW_Selected = sum(MW * is_selected,na.rm = TRUE),
#   )


}
}
}
}
}
}

# Change directory to patio econ results to store all subsequent results

if(user == "Cfong") {
  setwd(paste("~/RMI/Utility Transition Finance - Clean Repowering/Analysis/Economic Model Results",patio_results,sep="/"))
} else if (user == "udayvaradarajan") {
  uvwd = "/Users/udayvaradarajan/My Drive/GitHub/patio-model"
  # uvresultsdir = paste(uvwd,"/","econ_results/",patio_results,"/",sep="")
  uvresultsdir = paste("/Users/udayvaradarajan/Library/CloudStorage/OneDrive-RMI/Clean Repowering/Analysis/Economic Model Results/",patio_results,"/",sep = "")
  if(!dir.exists(uvresultsdir)) {dir.create(uvresultsdir)}
  setwd(uvresultsdir)
} else {
  "set file path for user"
}

# save results on the cloud

cloud$write_patio_econ_results(final_outputs, patio_results, "final_outputs.parquet")
cloud$write_patio_econ_results(scenario_selected, patio_results, "scenario_selected.parquet")

parameters <- toJSON(parameters)
cloud$write_patio_econ_results(parameters, patio_results, "parameters.json")


final_outputs_agg <- read_parquet("final_outputs.parquet")

year_to_test = 2032

final_outputs_by_plant <- final_outputs_agg %>%
  filter(!is.na(category)) %>%
  group_by(
    plant_id_eia,
    ba_code,
    refi_trigger,
    savings_tol,
    earnings_thresh,
    least_cost,
    Debt_Repl_EIR,
    Equity_Repl_EIR,
    counterfactual_baseline,
    historical_actuals
  ) %>%
  summarize(
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs), ~ sum(. * Disc * (operating_year %in% irp_years), na.rm = TRUE),.names = "{.col}_Disc_Sum"),
    MW = sum(MW * (Technology_FERC!="transmission") * (operating_year == year_to_test), na.rm = TRUE),
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs,Emissions,Capex,Securitization_Amount), ~ sum(. * (operating_year == year_to_test), na.rm = TRUE)),
    Cumulative_Emissions = sum(Emissions * (operating_year <= year_to_test), na.rm = TRUE),
    Cost_of_Electricity = if_else(MWh>0,Costs / MWh,NA),
    LCOE = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,NA),
    across(c(Capex_Costs_Disc_Sum,Fuel_Costs_Disc_Sum,FOM_Disc_Sum,VOM_Disc_Sum,Startup_Costs_Disc_Sum,Opex_Disc_Sum),
           ~ if_else(MWh_Disc_Sum>0, . / MWh_Disc_Sum,NA),.names = "L{.col}"),
    CF = MWh / (MW * (8760+if_else(year_to_test %% 4 == 0,24,0))),
    Counterfactual_Emissions_2021 = sum(Emissions * (operating_year == 2021) * counterfactual_baseline, na.rm = TRUE),
    Historical_Emissions_2021 = sum(Emissions * (operating_year == 2021) * historical_actuals, na.rm = TRUE)
  ) %>% ungroup()

final_outputs_by_category <- final_outputs_agg %>%
  filter(!is.na(category)) %>%
  group_by(
    refi_trigger,
    savings_tol,
    earnings_thresh,
    least_cost,
    Debt_Repl_EIR,
    Equity_Repl_EIR,
    counterfactual_baseline,
    historical_actuals,
    category
  ) %>%
  summarize(
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs), ~ sum(. * Disc * (operating_year %in% irp_years), na.rm = TRUE),.names = "{.col}_Disc_Sum"),
    MW = sum(MW * (Technology_FERC!="transmission") * (operating_year == year_to_test), na.rm = TRUE),
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs,Emissions,Capex,Securitization_Amount), ~ sum(. * (operating_year == year_to_test), na.rm = TRUE)),
    Cumulative_Emissions = sum(Emissions * (operating_year <= year_to_test), na.rm = TRUE),
    Cost_of_Electricity = if_else(MWh>0,Costs / MWh,NA),
    LCOE = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,NA),
    across(c(Capex_Costs_Disc_Sum,Fuel_Costs_Disc_Sum,FOM_Disc_Sum,VOM_Disc_Sum,Startup_Costs_Disc_Sum,Opex_Disc_Sum),
           ~ if_else(MWh_Disc_Sum>0, . / MWh_Disc_Sum,NA),.names = "L{.col}"),
    CF = MWh / (MW * (8760+if_else(year_to_test %% 4 == 0,24,0))),
    Counterfactual_Emissions_2021 = sum(Emissions * (operating_year == 2021) * counterfactual_baseline, na.rm = TRUE),
    Historical_Emissions_2021 = sum(Emissions * (operating_year == 2021) * historical_actuals, na.rm = TRUE)
  ) %>% ungroup()

final_outputs_by_category_technology <- final_outputs_agg %>%
  filter(!is.na(category)) %>%
  group_by(
    refi_trigger,
    savings_tol,
    earnings_thresh,
    least_cost,
    Debt_Repl_EIR,
    Equity_Repl_EIR,
    counterfactual_baseline,
    historical_actuals,
    category,
    technology_description
  ) %>%
  summarize(
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs), ~ sum(. * Disc * (operating_year %in% irp_years), na.rm = TRUE),.names = "{.col}_Disc_Sum"),
    MW = sum(MW * (Technology_FERC!="transmission") * (operating_year == year_to_test), na.rm = TRUE),
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs,Emissions,Capex,Securitization_Amount), ~ sum(. * (operating_year == year_to_test), na.rm = TRUE)),
    Cumulative_Emissions = sum(Emissions * (operating_year <= year_to_test), na.rm = TRUE),
    Cost_of_Electricity = if_else(MWh>0,Costs / MWh,NA),
    LCOE = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,NA),
    across(c(Capex_Costs_Disc_Sum,Fuel_Costs_Disc_Sum,FOM_Disc_Sum,VOM_Disc_Sum,Startup_Costs_Disc_Sum,Opex_Disc_Sum),
           ~ if_else(MWh_Disc_Sum>0, . / MWh_Disc_Sum,NA),.names = "L{.col}"),
    CF = MWh / (MW * (8760+if_else(year_to_test %% 4 == 0,24,0))),
    Counterfactual_Emissions_2021 = sum(Emissions * (operating_year == 2021) * counterfactual_baseline, na.rm = TRUE),
    Historical_Emissions_2021 = sum(Emissions * (operating_year == 2021) * historical_actuals, na.rm = TRUE)
  ) %>% ungroup()


final_outputs_summary <- final_outputs_agg %>%
  filter(!is.na(category)) %>%
  group_by(
    refi_trigger,
    savings_tol,
    earnings_thresh,
    least_cost,
    Debt_Repl_EIR,
    Equity_Repl_EIR,
    counterfactual_baseline,
    historical_actuals#,
    # category
  ) %>%
  summarize(
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs), ~ sum(. * Disc * (operating_year %in% irp_years), na.rm = TRUE),.names = "{.col}_Disc_Sum"),
    MW = sum(MW * (Technology_FERC!="transmission") * (operating_year == year_to_test), na.rm = TRUE),
    across(c(MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Costs,Emissions,Capex,Securitization_Amount), ~ sum(. * (operating_year == year_to_test), na.rm = TRUE)),
    Cumulative_Emissions = sum(Emissions * (operating_year <= year_to_test), na.rm = TRUE),
    Cost_of_Electricity = if_else(MWh>0,Costs / MWh,NA),
    LCOE = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,NA),
    across(c(Capex_Costs_Disc_Sum,Fuel_Costs_Disc_Sum,FOM_Disc_Sum,VOM_Disc_Sum,Startup_Costs_Disc_Sum,Opex_Disc_Sum),
           ~ if_else(MWh_Disc_Sum>0, . / MWh_Disc_Sum,NA),.names = "L{.col}"),
    Counterfactual_Emissions_2021 = sum(Emissions * (operating_year == 2021) * counterfactual_baseline, na.rm = TRUE),
    Historical_Emissions_2021 = sum(Emissions * (operating_year == 2021) * historical_actuals, na.rm = TRUE),
  ) %>% ungroup()

final_outputs_by_ba_code <- final_outputs_agg %>%
  filter(!is.na(category),!historical_actuals) %>%
  group_by(
    ba_code,
    refi_trigger,
    savings_tol,
    earnings_thresh,
    least_cost,
    Debt_Repl_EIR,
    Equity_Repl_EIR
    #category,
    #technology_description
  ) %>%
  summarize(
    across(c(Costs,MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex), ~ sum(. * Disc * (operating_year %in% irp_years) * (!counterfactual_baseline), na.rm = TRUE),.names = "{.col}_Disc_Sum"),
    across(c(Costs,MWh,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex), ~ sum(. * Disc * (operating_year %in% irp_years) * counterfactual_baseline, na.rm = TRUE),.names = "{.col}_Disc_Sum_count"),
    LCOE = if_else(MWh_Disc_Sum>0,Costs_Disc_Sum/MWh_Disc_Sum,NA),
    LCOE_count = if_else(MWh_Disc_Sum_count>0,Costs_Disc_Sum_count/MWh_Disc_Sum_count,NA),
    LCOE_Diff = LCOE_count - LCOE,
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex), ~ sum(. * Disc * (operating_year %in% irp_years) * (2*counterfactual_baseline-1), na.rm = TRUE),.names = "{.col}_Disc_Sum_Diff"),
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex,Emissions,Capex,Securitization_Amount), ~ sum(. * (operating_year == year_to_test) * (2*counterfactual_baseline-1), na.rm = TRUE),.names = "{.col}_Diff"),
    across(c(Costs,Capex_Costs,Fuel_Costs,FOM,VOM,Startup_Costs,Opex), ~ sum(. * Disc * (operating_year %in% irp_years) * (2*counterfactual_baseline-1), na.rm = TRUE),.names = "{.col}_Disc_Sum_Diff"),
  ) %>% ungroup()



# file.remove("final_data_index.parquet")
file.remove("allocated_parquet.parquet")


# final_fossil_costs_name <- paste("final_fossil_costs-",patio_results,".parquet",sep="")
# final_prop_and_clean_outputs_name <- paste("final_prop_and_clean_data-",patio_results,".parquet",sep="")
# final_fossil_outputs_name <- paste("final_fossil_data-",patio_results,".parquet",sep="")
# final_refinancing_costs_name <- paste("final_refinancing_costs-",patio_results,".parquet",sep="")
# final_refinancing_outputs_name <- paste("final_refinancing_outputs-",patio_results,".parquet",sep="")
# parquet_name <- paste("final_econ_by_tech-",patio_results,".parquet",sep="")
# write_parquet(final_fossil_costs,final_fossil_costs_name)
# write_parquet(final_fossil_outputs,final_fossil_outputs_name)
# write_parquet(final_refinancing_costs,final_refinancing_costs_name)
# write_parquet(final_refinancing_outputs,final_refinancing_outputs_name)
# write_parquet(final_prop_and_clean_outputs,final_prop_and_clean_outputs_name)
# write_parquet(final_econ_by_tech,parquet_name)
# write.csv(final_econ_by_tech %>% filter(is_selected), paste("final_econ_by_tech-",patio_results,".csv",sep=""))
# write.csv(final_econ_by_category %>% filter(is_selected), paste("final_econ_by_category-",patio_results,".csv",sep=""))
# write.csv(final_econ %>% filter(is_selected), paste("final_econ-",patio_results,".csv",sep=""))

# # Calculate project financed alternative costs for clean assets with tax equity if required
#
# final_data_inputs_PF <- final_data_inputs %>%
#   select(c(
#     index,
#     cur_project_year,
#     build_year,
#     re_plant_id,
#     Utility_ID,
#     can_claim_credit,
#     can_monetize_credit,
#     direct_pay_eligible,
#     transferability_eligible,
#     tax_equity_used,
#     scenario,
#     energy_community,
#     plant_id_eia,
#     generator_id,
#     ba_code,
#     re_energy,
#     nuclear_scen,
#     storage_li_pct,
#     entity_type,
#     technology_description,
#     ccs_eligible,
#     ccs_scen,
#     Inflator,
#     Clean_CapEx,
#     Clean_PTC,
#     Clean_45Q,
#     Clean_OpEx,
#     ITC,
#     roe,
#     cur_MACRS,
#     sum_MACRS,
#     MW,
#     MWh
#   )) %>%
#   left_join(Tax_Equity_Params, by = c("technology_description" = "Technology")) %>%
#   group_by(index) %>%
#   arrange(cur_project_year) %>%
#   mutate(
#
#     # Start with calculation of dummy variables and fractions of tax and cost allocations between tax investors,
#     # equity investor, and (back-levered) debt in each year. These tax equity parameters will differ for assets
#     # using an ITC vs. the PTC.
#
#     PF_Equity_Hurdle = tax_equity_used * Equity_Hurdle + (!tax_equity_used) * roe,
#     PF_Debt_Cost = tax_equity_used * (Debt_Spread + Forward_Treasury_Rates) + (!tax_equity_used) * debt_cost,
#     ptc_year = (cur_project_year <= 10),
#     ITC_pre_flip = (cur_project_year<=ITC_Flip),
#     PTC_pre_flip = (cur_project_year<=PTC_Flip),
#     ITC_tax_basis_mod_year = ((cur_project_year>1) & (cur_project_year<=ITC_Basis_Mod_Year)),
#     ITC_CRP_year = (cur_project_year<=ITC_CRP),
#     debt_year = (cur_project_year<=PF_Debt_Tenor),
#     ITC_TE_Frac_Tax = (ITC_pre_flip *
#                          (ITC_TE_tax_basis_mod * ITC_tax_basis_mod_year + (!ITC_tax_basis_mod_year) * ITC_TE_tax_pre) +
#                          (!ITC_pre_flip) * ITC_TE_tax_post),
#     ITC_TE_Frac_Cash = (ITC_pre_flip * (!ITC_CRP_year) * ITC_TE_cash_pre +
#                           (!ITC_pre_flip) * ITC_TE_cash_post),
#     CE_Frac = (debt_year * (Debt_DSCR-1)/Debt_DSCR+(!debt_year)),
#     ITC_CE_Frac = (debt_year * (ITC_Debt_DSCR-1)/ITC_Debt_DSCR+(!debt_year)),
#     PTC_TE_Frac_Tax = (PTC_pre_flip * PTC_TE_tax_pre + (!PTC_pre_flip) * PTC_TE_tax_post),
#     PTC_TE_Frac_Cash = (PTC_pre_flip * PTC_TE_cash_pre + (!PTC_pre_flip) * PTC_TE_cash_post),
#     PTC_CE_Frac = (debt_year * (PTC_Debt_DSCR-1)/PTC_Debt_DSCR+(!debt_year)),
#
#     # Now, compute the impacts of various operating costs and tax benefits on the cash and tax flows to various investors
#     # given the flip structure allocations defined above (the _Indep variables) and the coefficients of the impact of
#     # potential PPA revenues (the _Coeff variables) on those cash flows. The goal of this section is to then use those
#     # two sets of variables to solve for the level nominal PPA price (measured as a fraction of initial CAPEX) needed
#     # to meet the IRR hurdles for each of the investors (TE - tax equity, CE - cash equity, and Debt) simultaneously.
#
#     PF_Tax_Credits = first_year * ITC +
#       ptc_year * if_else(MW > 0, Clean_PTC / Clean_CapEx,0),
#     PF_OpEx = -if_else(MW > 0, Clean_OpEx / Clean_CapEx,0),
#     PF_OpEx_Tax = -PF_OpEx * PF_Tax_Rate,
#     PF_Pre_Flip = tax_equity_used * if_else(ITC>0,ITC_pre_flip,PTC_pre_flip),
#     CE_Frac_Interest_Tax = -(1-1/(1+PF_Debt_Cost)^cur_project_year),
#     PF_TE_Frac_Tax = tax_equity_used * if_else(ITC>0,ITC_TE_Frac_Tax,PTC_TE_Frac_Tax),
#     PF_TE_Frac_Cash = tax_equity_used * if_else(ITC>0,ITC_TE_Frac_Cash,PTC_TE_Frac_Cash),
#     PF_CE_Frac_Tax = (1-PF_TE_Frac_Tax) * if_else(tax_equity_used,if_else(ITC>0,ITC_CE_Frac,PTC_CE_Frac),CE_Frac),
#     PF_CE_Frac_Cash = (1-PF_TE_Frac_Cash) * if_else(tax_equity_used,if_else(ITC>0,ITC_CE_Frac,PTC_CE_Frac),CE_Frac),
#     PF_Debt_Frac_Tax = (1-PF_TE_Frac_Tax-PF_CE_Frac_Tax),
#     PF_Debt_Frac_Cash = (1-PF_TE_Frac_Cash-PF_CE_Frac_Cash),
#     PF_CE_Interest_Tax_Coeff = (PF_Debt_Frac_Tax+PF_Debt_Frac_Cash) * CE_Frac_Interest_Tax,
#     Debt_PPA_Indep_Tax = PF_Debt_Frac_Tax *
#       (tax_equity_used * PF_Tax_Credits + PF_MACRS_TE * PF_Tax_Rate + PF_OpEx_Tax),
#     Debt_PPA_Coeff_Tax = -PF_Tax_Rate * PF_Debt_Frac_Tax,
#     CE_PPA_Indep_Tax = PF_CE_Frac_Tax *
#       (tax_equity_used * PF_Tax_Credits + PF_MACRS_TE * PF_Tax_Rate + PF_OpEx_Tax) -
#       (Debt_PPA_Indep_Tax + (PF_Debt_Frac_Tax * (!tax_equity_used) * PF_Tax_Credits +
#                                PF_Debt_Frac_Cash * PF_OpEx)) *
#       CE_Frac_Interest_Tax * PF_Tax_Rate,
#     CE_PPA_Coeff_Tax = -PF_Tax_Rate * (PF_CE_Frac_Tax + PF_CE_Interest_Tax_Coeff),
#     TE_PPA_Indep_Disc = PF_Pre_Flip * (PF_TE_Frac_Tax * (PF_Tax_Credits + PF_MACRS_TE * PF_Tax_Rate + PF_OpEx_Tax) +
#                                          PF_TE_Frac_Cash * PF_OpEx) / (1+TE_Yield)^cur_project_year,
#     TE_PPA_Coeff_Disc = PF_Pre_Flip * (-PF_Tax_Rate * PF_TE_Frac_Tax + PF_TE_Frac_Cash) / (1+TE_Yield)^cur_project_year,
#     Debt_PPA_Indep_Cash_Disc = (PF_Debt_Frac_Tax * (!tax_equity_used) * PF_Tax_Credits +
#                                   PF_Debt_Frac_Cash * PF_OpEx) / (1+PF_Debt_Cost)^cur_project_year,
#     Debt_PPA_Coeff_Cash_Disc = PF_Debt_Frac_Cash / (1+PF_Debt_Cost)^cur_project_year,
#     CE_PPA_Indep_Cash_Disc = (PF_CE_Frac_Tax * (!tax_equity_used) * PF_Tax_Credits +
#                                 PF_CE_Frac_Cash * PF_OpEx) / (1+PF_Equity_Hurdle)^cur_project_year,
#     CE_PPA_Coeff_Cash_Disc = PF_CE_Frac_Cash / (1+PF_Equity_Hurdle)^cur_project_year,
#     Debt_PPA_Indep_Tax_Disc = Debt_PPA_Indep_Tax / (1+PF_Debt_Cost)^cur_project_year,
#     Debt_PPA_Coeff_Tax_Disc = Debt_PPA_Coeff_Tax / (1+PF_Debt_Cost)^cur_project_year,
#     CE_PPA_Indep_Tax_Disc = CE_PPA_Indep_Tax / (1+PF_Equity_Hurdle)^cur_project_year,
#     CE_PPA_Coeff_Tax_Disc = CE_PPA_Coeff_Tax / (1+PF_Equity_Hurdle)^cur_project_year,
#     TE_Indep_Disc_Sum = cumsum(TE_PPA_Indep_Disc),
#     TE_Coeff_Disc_Sum = cumsum(TE_PPA_Coeff_Disc),
#     CE_Indep_Cash_Disc_Sum = cumsum(CE_PPA_Indep_Cash_Disc),
#     CE_Coeff_Cash_Disc_Sum = cumsum(CE_PPA_Coeff_Cash_Disc),
#     Debt_Indep_Cash_Disc_Sum = cumsum(Debt_PPA_Indep_Cash_Disc),
#     Debt_Coeff_Cash_Disc_Sum = cumsum(Debt_PPA_Coeff_Cash_Disc),
#     Debt_Indep_Tax_Sum = cumsum(Debt_PPA_Indep_Tax),
#     Debt_Coeff_Tax_Sum = cumsum(Debt_PPA_Coeff_Tax),
#     CE_Indep_Tax_Sum = cumsum(CE_PPA_Indep_Tax),
#     CE_Coeff_Tax_Sum = cumsum(CE_PPA_Coeff_Tax),
#     Debt_Indep_Tax_Disc_Sum = cumsum(Debt_PPA_Indep_Tax_Disc),
#     Debt_Coeff_Tax_Disc_Sum = cumsum(Debt_PPA_Coeff_Tax_Disc),
#     CE_Indep_Tax_Disc_Sum = cumsum(CE_PPA_Indep_Tax_Disc),
#     CE_Coeff_Tax_Disc_Sum = cumsum(CE_PPA_Coeff_Tax_Disc)
#   )
#
#
#
#
#
# ### NOTES
# # Long duration storage costs needs; maybe CCS or hydrogen? in system_parquet
# # reach out to David V and Tessa
# # options: drop it, change it to long duration storage, or get a better cost estimate
# ## think about how to calculate transmission distances + weights
# ### nuclear CF not addressable/super impactful — variable O&M/fuel costs not huge
# #### worry less about percentage, look more at MW of wind, solar, storage
# #### $ / GW * new build GW needed; integrate wind/solar or keep separate?
# #### maybe try aggregating to NREL class?
#
#
#
#
#

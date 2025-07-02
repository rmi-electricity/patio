# Utilities Inputs
library(lubridate)
library(dplyr)
library(lmtest)
library(ggplot2)
library(broom)
library(tidyr)
library(janitor)

options(show.error.locations = TRUE)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1) {
  patio_results <- args
} else {
  warning("No patio result datestr supplied")
  patio_results <- "202504270143"
}
print(paste("Creating utility inputs for", patio_results))
junk_pattern <- "[_&#$\\.,:;\\*/@]|\\(\\w*\\)|llc"
utility_delim <- "[\\.,:;]|\\scooperative|\\scoop|\\sinc|\\sco-op"
delim_string <- paste(
  "[_&#$\\.,:;\\*/@]|\\(\\w*\\)|llc",
  "no\\.|\\sunits|\\sand\\s|\\sunit|subtotal|total|un\\.|com\\.|common|\\splant|\\spower|steam-|scr|fgd",
  "\\spipeline|\\spipes|pipe|\\sun\\s|\\srail|power|\\sgenerator|\\senergy|\\senergy|\\sportion|\\sshare",
  "\\scenter|\\sctr|\\shybrid|\\speaking|\\sheat|\\sproject|\\sgeneration|\\sgenerating|\\scogeneration",
  "\\sfacility|\\smanufacturing|\\scogen|\\sllc|\\scycle|\\scombustion|\\scombined|\\sturbine|\\scycle",
  "pwr|steam|\\scoal\\s|\\scars|station|\\smw|\\seach|\\sgen\\.|\\scom\\s|\\sscr|\\snox|\\ssite|\\sferc|\\sfpc|\\sproject",
  sep = "|"
)
IRA_on <- TRUE

# python integration instructions here: https://rmi.github.io/etoolbox/etb_and_r.html#setup-etoolbox-with-uv
reticulate::use_python(gsub("lib/R", "bin/python", R.home()))
platformdirs <- reticulate::import("platformdirs")
constants <- reticulate::import("patio.constants")
cloud <- reticulate::import("etoolbox.utils.cloud")
cache_dir <- platformdirs$user_cache_dir("patio", ensure_exists = TRUE)
root_dir <- paste(constants$ROOT_PATH) # paste converts py_path to string
results <- cloud$read_patio_resource_results(patio_results)

# Define functions for string cleaning and fuzzy matching and extraction

# This is a vectorized, string-level cleaning function for plant names using a specified default vector of delimiters (delim_replace)
clean_name_string <- Vectorize(function(
  string_to_clean,
  delim_string,
  strip_all_numbers = FALSE,
  strip_large_numbers = FALSE
) {
  clean_name <- stringr::str_remove_all(tolower(string_to_clean), delim_string)
  if (is.na(clean_name) | clean_name == "") {
    return(clean_name)
  } else {
    # This function replaces "-" used to indicate a range of units with each unit number to facilitate matching, then using a flexible regexp search with the
    # upper and lower bounds around "-" with arbitrary spaces "\\s*" to replace them.
    lower_range <- as.integer(stringr::str_remove(
      stringr::str_extract(clean_name, "\\d+\\s*-"),
      "-"
    ))
    upper_range <- as.integer(stringr::str_remove(
      stringr::str_extract(clean_name, "-\\d+\\s*"),
      "-"
    ))
    if (
      !(is.na(lower_range) | is.na(upper_range) | lower_range >= upper_range)
    ) {
      cur_str <- paste(lower_range, "-", upper_range, sep = "\\s*")
      temp_str <- as.character(lower_range)
      for (unitnum in (lower_range + 1):upper_range) {
        temp_str <- paste(temp_str, unitnum, sep = " ")
      }
      clean_name <- stringr::str_replace(clean_name, cur_str, temp_str)
    }
    clean_name <- gsub("-", " ", clean_name, fixed = TRUE)
    clean_name <- gsub("u(\\d+)", "\\1", clean_name)
    clean_name <- trimws(gsub("\\s+", " ", clean_name))
  }
  if (strip_all_numbers) {
    clean_name <- stringr::str_remove_all(clean_name, "[0-9%]")
  }
  if (strip_large_numbers) {
    clean_name <- stringr::str_remove_all(clean_name, "[1-9]\\d{2,}")
  }
  clean_name <- trimws(gsub("\\s+", " ", clean_name))
  return(as.character(clean_name))
})

# This function generates an acronym out of the plant name
acron <- Vectorize(function(string_to_acron) {
  acron_found <- stringr::str_remove_all(tolower(string_to_acron), junk_pattern)
  acron_found <- stringr::str_replace_all(
    acron_found,
    "([a-zi+v*])\\w*\\s*",
    "\\1"
  )
  return(acron_found)
})

# This function measures the extent to which one particular sequence needs to be modified to "fit in" to another sequence.
# This is calculated as a modified levenshtein distance that more heavily weights insertions and substitutions relative to deletions,
# and has been empirically verified to provide high-quality matches between ferc and eia plant name fields.
string_fit <- Vectorize(function(S1, S2) {
  dweight <- 1 / stringr::str_length(S1)
  iweight <- 0.5
  sweight <- 1
  maxlen <- (2 *
    stringr::str_length(S1) *
    iweight +
    stringr::str_length(S2) * dweight)
  string_fit <- ifelse(
    maxlen > 0 & !is.na(maxlen),
    1 -
      stringdist::stringdist(
        S1,
        S2,
        method = "dl",
        weight = c(dweight, iweight, sweight, 1)
      ) /
        maxlen,
    1
  )
  return(string_fit)
})

# This function extracts large (>=100) numbers from a string in order to pull out ferc plant codes for hydro assets
extract_large_num <- Vectorize(function(string_to_check) {
  large_num <- as.integer(stringr::str_extract(string_to_check, "[1-9]\\d{2,}"))
  return(large_num)
})

# End function definitions

# Pull in model inputs relevant to utility capital costs and inflation as well as FERC-EIA matching based on Hub data
utilities_inputs <- cloud$read_cloud_file(
  "patio-data/20241031/utility_information.parquet.gzip"
)
utilities_inputs <- as.data.frame(
  unclass(utilities_inputs),
  stringsAsFactors = TRUE,
  optional = TRUE
) %>%
  select(-c(entity_type_eia, utility_type_rmi, state))
bbb_file <- paste(
  root_dir,
  "BBB Fossil Transition Analysis Inputs.xlsm",
  sep = "/"
)
model_inputs <- tidyxl::xlsx_names(bbb_file) %>%
  subset(is_range == TRUE & hidden == FALSE, select = c(name, formula))

named_ranges <- model_inputs$name
for (range_name in named_ranges) {
  range_formula <- model_inputs %>%
    subset(name == range_name, select = "formula") %>%
    as.character()
  if (!grepl(":", range_formula, fixed = TRUE)) {
    range_value_test <- readxl::read_excel(
      bbb_file,
      range = range_formula,
      col_names = FALSE
    )
    range_value <- ifelse(
      is.na(as.numeric(range_value_test)),
      as.character(range_value_test),
      as.numeric(range_value_test)
    )
    assign(range_name, range_value)
  } else if (
    grepl("Toggles", range_formula, fixed = TRUE) |
      grepl("Tax Depreciation", range_formula, fixed = TRUE) |
      grepl("BLS Data Series", range_formula, fixed = TRUE) |
      grepl("Interest Rates", range_formula, fixed = TRUE)
  ) {
    range_value_test <- readxl::read_excel(bbb_file, range = range_formula)
    range_value_test <- as.data.frame(
      unclass(range_value_test),
      stringsAsFactors = TRUE,
      optional = TRUE
    )
    assign(range_name, range_value_test)
  }
}
CPIU <- CPIU[c("Year", "Inflation_Factor_2021")]
fuel_map <- cloud$read_cloud_file(
  "patio-data/20241031/fuel_group_and_emissions_map.parquet"
) %>%
  select(c(energy_source_code, fuel_group_code))

# Pull in state tax data
state_mapping <- cloud$read_cloud_file(
  "patio-data/20241031/state_re_fraction_class.parquet"
) %>%
  filter(!is.na(State)) %>%
  mutate(
    State_Tax_Rate = case_when(
      is.na(State_Tax_Rate) ~ 0,
      TRUE ~ State_Tax_Rate
    )
  )
state_mapping <- as.data.frame(
  unclass(state_mapping),
  stringsAsFactors = TRUE,
  optional = TRUE
)

### Pull in resource model data
full_parquet_map <- results$full %>%
  select(
    plant_id_eia,
    prime_mover_code,
    generator_id,
    # sector_name,
    capacity = capacity_mw,
    State = state,
    Technology = technology_description,
    # energy_source_code = energy_source_code_1,
    fuel_group_code = fuel_group,
    utility_id_eia,
    # td_utility_id_eia,
    # Technology_FERC,
    operational_status
  ) %>%
  distinct() %>%
  mutate(
    sector_name = NA,
    energy_source_code = NA,
    td_utility_id_eia = utility_id_eia,
    Technology_FERC = case_when(
      grepl("Nuclear", Technology) ~ "nuclear",
      grepl("Battery", Technology) ~ "renewables",
      grepl("Hydro", Technology) ~ "hydro",
      (grepl("Natural Gas", Technology) | grepl("Coal", Technology)) &
        grepl("Steam", Technology) ~
        "steam",
      (grepl("Natural Gas", Technology) |
        grepl("Coal", Technology) |
        grepl("Solid", Technology) |
        grepl("Petroleum", Technology) |
        grepl("Gas", Technology) |
        grepl("All", Technology)) ~
        "other_fossil",
      TRUE ~ "renewables"
    )
  )

reg_lag <- 3

# Pull in PUDL utility ID matching tables to use for subsequent calculations, and normalize all ferc IDs to utility_id_ferc1,
# incorporating additional FERC / EIA utility matches developed in Hub / Patio modeling as recorded in "utilities_inputs.xlsx"
pudl_release <- results$pudl_release
pudl <- reticulate::import("etoolbox.utils.pudl")
utilities_eia <- pudl$pd_read_pudl(
  "core_pudl__assn_eia_pudl_utilities",
  release = pudl_release
)
utilities_ferc1 <- pudl$pd_read_pudl(
  "core_pudl__assn_ferc1_pudl_utilities",
  release = pudl_release
)
utilities_ferc1_dbf <- pudl$pd_read_pudl(
  "core_pudl__assn_ferc1_dbf_pudl_utilities",
  release = pudl_release
)
utilities_inputs <- utilities_inputs %>%
  left_join(
    utilities_ferc1_dbf,
    by = c("respondent_id" = "utility_id_ferc1_dbf")
  )
utilities_ferc1 <- utilities_ferc1 %>%
  left_join(
    utilities_eia,
    by = c("utility_id_pudl"),
    relationship = "many-to-many"
  )

# Operational data (861)
operational_data_misc_eia861 <- pudl$pd_read_pudl(
  "core_eia861__yearly_operational_data_misc",
  release = pudl_release
) %>%
  filter(year(report_date) == 2022) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>%
  mutate(
    total_sources = net_generation_mwh +
      wholesale_power_purchases_mwh +
      net_power_exchanged_mwh +
      net_wheeled_power_mwh
  )
operational_data_revenue_eia861 <- pudl$pd_read_pudl(
  "core_eia861__yearly_operational_data_revenue",
  release = pudl_release
) %>%
  filter(year(report_date) == 2022) %>%
  pivot_wider(names_from = revenue_class, values_from = revenue) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>%
  mutate(
    total_revenue = credits_or_adjustments +
      delivery_customers +
      other +
      retail_sales +
      sales_for_resale +
      transmission
  )

operational_data_eia861 <- operational_data_misc_eia861 %>%
  left_join(
    operational_data_revenue_eia861,
    by = c(
      "utility_id_eia",
      "nerc_region",
      "state",
      "report_date",
      "data_maturity"
    )
  )
operational_data_eia861 <- operational_data_eia861 %>%
  mutate(
    frac_retail_sales_mwh = retail_sales_mwh /
      (retail_sales_mwh + sales_for_resale_mwh),
    frac_sales_for_resale_mwh = 1 - frac_retail_sales_mwh,
    frac_GT = (transmission + sales_for_resale) / total_revenue,
    frac_own_gen = net_generation_mwh / total_sources
  )

# Sales to Ultimate Customers (861)
sales_eia861 <- pudl$pd_read_pudl(
  "core_eia861__yearly_sales",
  release = pudl_release
) %>%
  filter(year(report_date) == 2022) %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0))) %>%
  pivot_wider(
    names_from = customer_class,
    values_from = c(customers, sales_mwh, sales_revenue)
  ) %>%
  mutate(
    total_revenue = sales_revenue_commercial +
      sales_revenue_industrial +
      sales_revenue_other +
      sales_revenue_residential +
      sales_revenue_transportation,
    total_sales_mwh = sales_mwh_commercial +
      sales_mwh_industrial +
      sales_mwh_other +
      sales_mwh_residential +
      sales_mwh_transportation
  )
sales_eia861 <- sales_eia861 %>%
  left_join(
    state_mapping %>%
      select(c("State", "State_Tax_Rate")),
    by = c("state" = "State")
  )
sales_eia861_summary <- sales_eia861 %>%
  group_by(utility_id_eia) %>%
  summarize(
    cust_revenue = sum(total_revenue),
    cust_sales_mwh = sum(total_sales_mwh),
    Average_State_Tax_Rate = sum(State_Tax_Rate * total_revenue, na.rm = TRUE) /
      cust_revenue,
    com_frac_revenue = sum(sales_revenue_commercial, na.rm = TRUE) /
      cust_revenue,
    ind_frac_revenue = sum(sales_revenue_industrial, na.rm = TRUE) /
      cust_revenue,
    other_frac_revenue = sum(sales_revenue_other, na.rm = TRUE) / cust_revenue,
    res_frac_revenue = sum(sales_revenue_residential, na.rm = TRUE) /
      cust_revenue,
    trans_frac_revenue = sum(sales_revenue_transportation, na.rm = TRUE) /
      cust_revenue,
    com_frac_sales_mwh = sum(sales_mwh_commercial, na.rm = TRUE) /
      cust_sales_mwh,
    ind_frac_sales_mwh = sum(sales_mwh_industrial, na.rm = TRUE) /
      cust_sales_mwh,
    other_frac_sales_mwh = sum(sales_mwh_other, na.rm = TRUE) / cust_sales_mwh,
    res_frac_sales_mwh = sum(sales_mwh_residential, na.rm = TRUE) /
      cust_sales_mwh,
    trans_frac_sales_mwh = sum(sales_mwh_transportation, na.rm = TRUE) /
      cust_sales_mwh
  )

# Utility Data (861)
utility_data_misc_eia861 <- pudl$pd_read_pudl(
  "core_eia861__yearly_utility_data_misc",
  release = pudl_release
) %>%
  filter(year(report_date) == 2022) %>%
  rename(ownership_type = entity_type) %>%
  left_join(Ownership_Entity_Table, by = c("ownership_type")) %>%
  mutate(
    entity_type_861 = as.factor(entity_type)
    # generation_activity = as.numeric(if_else((is.na(generation_activity) & !is.na(entity_type_861)) | generation_activity=="N",0,1)),
    # transmission_activity = as.numeric(if_else((is.na(transmission_activity) & !is.na(entity_type_861)) | transmission_activity=="N",0,1)),
    # distribution_activity = as.numeric(if_else((is.na(distribution_activity) & !is.na(entity_type_861)) | distribution_activity=="N",0,1))
  ) %>%
  select(-c(entity_type))

# Ownership (860)
ownership_eia860 <- pudl$pd_read_pudl(
  "core_eia860__scd_ownership",
  release = pudl_release
) %>%
  filter(year(report_date) == 2022) %>%
  mutate(
    generator_id = stringr::str_remove(generator_id, "^0000"),
    generator_id = stringr::str_remove(generator_id, "^000"),
    generator_id = stringr::str_remove(generator_id, "^00"),
    generator_id = stringr::str_remove(generator_id, "^0"),
  )

# Utility Data (860)
utilities_eia860 <- pudl$pd_read_pudl(
  "core_eia860__scd_utilities",
  release = pudl_release
) %>%
  arrange(utility_id_eia) %>%
  group_by(utility_id_eia) %>%
  mutate(
    has_entity_info = sum(!is.na(entity_type)),
    is_newest = (year(report_date) ==
      max(
        year(report_date) *
          ((has_entity_info > 0 & !is.na(entity_type)) | has_entity_info == 0),
        na.rm = TRUE
      )),
    entity_type = as.factor(entity_type)
  ) %>%
  filter(
    (has_entity_info == 0 & is_newest) |
      (is_newest & has_entity_info > 0 & !is.na(entity_type))
  ) %>%
  select(-c(has_entity_info)) %>%
  ungroup()

# Plants and Entity Data (860)
eia_plant_data <- pudl$pd_read_pudl(
  "core_eia__entity_plants",
  release = pudl_release
) %>%
  select(plant_id_eia, city, county, plant_name_eia) %>%
  mutate(
    clean_plant_acron = acron(plant_name_eia),
    clean_plant_name_eia = clean_name_string(plant_name_eia, delim_string),
    county = tolower(county),
    city = tolower(city)
  )
plants_eia860 <- pudl$pd_read_pudl(
  "core_eia860__scd_plants",
  release = pudl_release
) %>%
  arrange(plant_id_eia) %>%
  group_by(plant_id_eia) %>%
  mutate(
    has_entity_info = sum(
      !(is.na(sector_name_eia) & is.na(transmission_distribution_owner_id))
    ),
    is_newest = (year(report_date) ==
      max(
        year(report_date) *
          ((has_entity_info > 0 &
            !(is.na(sector_name_eia) &
              is.na(transmission_distribution_owner_id))) |
            has_entity_info == 0),
        na.rm = TRUE
      )),
    sector_name_eia = as.factor(sector_name_eia)
  ) %>%
  filter(
    (has_entity_info == 0 & is_newest) |
      (is_newest &
        has_entity_info > 0 &
        !(is.na(sector_name_eia) & is.na(transmission_distribution_owner_id)))
  ) %>%
  select(-c(has_entity_info)) %>%
  ungroup()

plants_entity_eia <- pudl$pd_read_pudl(
  "core_eia__entity_plants",
  release = pudl_release
)

generators_eia860 <- pudl$pd_read_pudl(
  "core_eia860__scd_generators",
  release = pudl_release
) %>%
  filter(year(report_date) >= 2023) %>%
  left_join(
    plants_entity_eia %>% select(c(plant_id_eia, State = state)),
    by = c("plant_id_eia")
  ) %>%
  left_join(
    plants_eia860 %>%
      select(c(
        plant_id_eia,
        sector_name = sector_name_eia,
        td_utility_id_eia = transmission_distribution_owner_id
      )),
    by = c("plant_id_eia")
  ) %>%
  left_join(fuel_map, by = c("energy_source_code_1" = "energy_source_code")) %>%
  mutate(
    td_utility_id_eia = coalesce(td_utility_id_eia, utility_id_eia),
    prime_mover_code_no_CC = prime_mover_code,
    prime_mover_code = if_else(
      prime_mover_code %in% c("CA", "CT", "CS"),
      "CC",
      prime_mover_code
    ),
    Technology_FERC = case_when(
      grepl("Nuclear", technology_description) ~ "nuclear",
      grepl("Battery", technology_description) ~ "renewables",
      grepl("Hydro", technology_description) ~ "hydro",
      (grepl("Natural Gas", technology_description) |
        grepl("Coal", technology_description)) &
        grepl("Steam", technology_description) ~
        "steam",
      (grepl("Natural Gas", technology_description) |
        grepl("Coal", technology_description) |
        grepl("Solid", technology_description) |
        grepl("Petroleum", technology_description) |
        grepl("Gas", technology_description) |
        grepl("All", technology_description)) ~
        "other_fossil",
      TRUE ~ "renewables"
    )
  )

transmission_capacity <- generators_eia860 %>%
  select(td_utility_id_eia, capacity_mw, operational_status) %>%
  filter(operational_status == "existing") %>%
  group_by(td_utility_id_eia) %>%
  summarize(capacity_mw = sum(capacity_mw, na.rm = TRUE))

transmission_owners_details <- plants_eia860 %>%
  select(
    td_utility_id_eia = transmission_distribution_owner_id,
    utility_id_eia,
  ) %>%
  mutate(td_utility_id_eia = coalesce(td_utility_id_eia, utility_id_eia)) %>%
  distinct() %>%
  arrange(td_utility_id_eia) %>%
  left_join(
    utilities_eia860 %>%
      select(c(utility_id_eia, entity_type_TD = entity_type)),
    by = c("td_utility_id_eia" = "utility_id_eia")
  ) %>%
  left_join(
    utility_data_misc_eia861 %>%
      select(c(utility_id_eia, entity_type_TD_861 = entity_type_861)),
    by = c("td_utility_id_eia" = "utility_id_eia")
  ) %>%
  mutate(entity_type_TD = coalesce(entity_type_TD, entity_type_TD_861)) %>%
  select(-c(entity_type_TD_861)) %>%
  distinct() %>%
  left_join(transmission_capacity, by = c("td_utility_id_eia"))

full_parquet_extra <- full_parquet_map %>%
  left_join(
    generators_eia860 %>%
      select(plant_id_eia, generator_id, energy_source_code_1),
    by = c("plant_id_eia", "generator_id")
  ) %>%
  filter(
    is.na(energy_source_code_1) & !is.na(utility_id_eia) & capacity > 0
  ) %>%
  select(-c(energy_source_code_1))

unit_level_data_pudl <- generators_eia860 %>%
  select(c(
    plant_id_eia,
    prime_mover_code,
    generator_id,
    sector_name,
    capacity = capacity_mw,
    State,
    Technology = technology_description,
    energy_source_code = energy_source_code_1,
    fuel_group_code,
    utility_id_eia,
    td_utility_id_eia,
    Technology_FERC,
    operational_status
  )) %>%
  arrange(plant_id_eia, generator_id)

tech_map_pudl <- unit_level_data_pudl %>%
  select(c(
    Technology,
    prime_mover_code,
    energy_source_code,
    fuel_group_code,
    Technology_FERC
  )) %>%
  distinct()

unit_level_data_pudl <- rbind(unit_level_data_pudl, full_parquet_extra)

# Build on PUDL's eia_match_table by
generators_eia860_for_matching <- generators_eia860 %>%
  select(c(
    utility_id_eia,
    plant_id_eia,
    generator_id,
    prime_mover_code,
    technology_description,
    energy_source_code_1,
    fuel_group_code,
    capacity_mw
  ))
plants_ferc1 <- pudl$pd_read_pudl(
  "out_ferc1__yearly_all_plants",
  release = pudl_release
)

# co-ops
coop_data <- cloud$read_cloud_file(
  "patio-data/20241031/co_op_analysis_summary.parquet"
) %>%
  select(c(
    Utility_ID,
    coop_utility_name,
    coop_type,
    coop_roe,
    coop_equity_ratio,
    coop_ror,
    coop_equity,
    coop_debt
  )) %>%
  distinct() %>%
  drop_na(Utility_ID)

coop_data <- coop_data %>%
  left_join(
    operational_data_eia861 %>%
      select(c(
        utility_id_eia,
        total_revenue,
        # frac_retail_sales_mwh,
        frac_sales_for_resale_mwh,
        # frac_GT,
        frac_own_gen
      )),
    by = c("Utility_ID" = "utility_id_eia")
  ) %>%
  mutate(
    coop_equity = coop_equity * 1000,
    coop_debt = coop_debt * 1000
  )

coop_data <- coop_data %>%
  left_join(
    utility_data_misc_eia861 %>%
      select(c(
        utility_id_eia,
        generation_activity,
        transmission_activity,
        distribution_activity,
        entity_type_861
      )),
    by = c("Utility_ID" = "utility_id_eia")
  )

coop_GT_Dist_Map <- cloud$read_cloud_file(
  "patio-data/20241031/co_op_map_xwalk.parquet"
) %>%
  mutate(
    GT_utility_name = if_else(
      GT_utility_name == "Continental Cooperative Services, Inc.",
      "South Mississippi El Pwr Assn",
      GT_utility_name
    )
  )

drop <- c("No G&T", "#NAME?", "")
GT_list <- coop_GT_Dist_Map %>%
  select(GT_utility_name) %>%
  distinct() %>%
  filter(!(GT_utility_name %in% drop)) %>%
  rbind(
    coop_GT_Dist_Map %>%
      select(GT_utility_name = GT_utility_name_1) %>%
      distinct() %>%
      filter(
        GT_utility_name %in%
          c("Associated Electric Co-op, Inc.", "Basin Electric Power Coop")
      )
    # coop_GT_Dist_Map %>% select(GT_utility_name = GT_utility_name_2) %>% distinct()
  ) %>%
  distinct() %>%
  mutate(
    clean_GT_name = trimws(gsub(
      "\\s+",
      " ",
      stringr::str_remove_all(tolower(GT_utility_name), utility_delim)
    ))
  ) %>%
  filter(clean_GT_name != "")

GT_match_list <- GT_list %>%
  cross_join(
    utilities_eia %>%
      left_join(
        utilities_eia860 %>% select(utility_id_eia, entity_type),
        by = c("utility_id_eia")
      ) %>%
      select(utility_name_eia, utility_id_eia, entity_type) %>%
      filter(entity_type == "C") %>%
      mutate(
        clean_EIA_name = trimws(gsub(
          "\\s+",
          " ",
          stringr::str_remove_all(tolower(utility_name_eia), utility_delim)
        ))
      )
  ) %>%
  group_by(clean_GT_name) %>%
  mutate(
    similarity = string_fit(clean_GT_name, clean_EIA_name),
    candidate = similarity >= (max(similarity, na.rm = TRUE) - 0.05),
    most_similar = similarity == max(similarity, na.rm = TRUE)
  )

GT_match <- GT_match_list %>%
  filter(most_similar & similarity > 0.86) %>%
  select(clean_GT_name, utility_name_eia, utility_id_eia) %>%
  filter(!(utility_id_eia %in% c(59159, 12915, 56621))) %>%
  distinct()

GT_list <- GT_list %>% left_join(GT_match, by = c("clean_GT_name"))

coop_GT_Dist_Map <- coop_GT_Dist_Map %>%
  left_join(
    GT_list %>%
      select(-c(clean_GT_name)) %>%
      rename(
        GT_utility_name_eia = utility_name_eia,
        GT_utility_id_eia = utility_id_eia
      ),
    by = c("GT_utility_name")
  ) %>%
  left_join(
    GT_list %>%
      select(-c(clean_GT_name)) %>%
      rename(
        SGT_utility_name_eia = utility_name_eia,
        SGT_utility_id_eia = utility_id_eia
      ),
    by = c("GT_utility_name_1" = "GT_utility_name")
  ) %>%
  left_join(
    sales_eia861_summary %>% select(utility_id_eia, cust_sales_mwh),
    by = c("dist_utility_id_eia" = "utility_id_eia")
  )

arrow::write_parquet(
  coop_GT_Dist_Map,
  paste(cache_dir, "coop_GT_Dist_Map.parquet", sep = "/")
)

coop_ba_codes <- c("531", "AECI", "556", "569", "58", "552")
coop_eia_ba_code_map <- as.data.frame(coop_ba_codes) %>%
  rename(ba_code = coop_ba_codes)
coop_eia_ba_code_map$SGT_utility_id_eia <- c(
  1307,
  924,
  5580,
  13994,
  7349,
  17568
)

coop_ba_code_remap <- coop_GT_Dist_Map %>%
  select(dist_utility_id_eia, GT_utility_id_eia, SGT_utility_id_eia) %>%
  mutate(
    SGT_utility_id_eia = coalesce(SGT_utility_id_eia, GT_utility_id_eia)
  ) %>%
  left_join(coop_eia_ba_code_map, by = c("SGT_utility_id_eia")) %>%
  filter(!is.na(ba_code))

arrow::write_parquet(
  coop_ba_code_remap,
  paste(cache_dir, "coop_ba_code_remap.parquet", sep = "/")
)

Coop_Equity_Ratio_Model <- lm(
  coop_equity_ratio ~
    frac_own_gen + generation_activity + frac_sales_for_resale_mwh,
  data = coop_data,
  weights = total_revenue
)
base_coop_er <- as.numeric(Coop_Equity_Ratio_Model$coefficients["(Intercept)"])
coeff_own_gen_coop_er <- as.numeric(Coop_Equity_Ratio_Model$coefficients[
  "frac_own_gen"
])
coeff_gen_coop_er <- as.numeric(Coop_Equity_Ratio_Model$coefficients[
  "generation_activity"
])
coeff_sales_coop_er <- as.numeric(Coop_Equity_Ratio_Model$coefficients[
  "frac_sales_for_resale_mwh"
])
summary(Coop_Equity_Ratio_Model)
Coop_ROR_Model <- lm(
  coop_ror ~ frac_own_gen,
  data = coop_data,
  weights = total_revenue
)
base_coop_ror <- as.numeric(Coop_ROR_Model$coefficients["(Intercept)"])
coeff_own_gen_coop_ror <- as.numeric(Coop_ROR_Model$coefficients[
  "frac_own_gen"
])
summary(Coop_ROR_Model)
Coop_ROE_Model <- lm(
  coop_roe ~ frac_own_gen,
  data = coop_data,
  weights = total_revenue
)
base_coop_roe <- as.numeric(Coop_ROE_Model$coefficients["(Intercept)"])
coeff_own_gen_coop_roe <- as.numeric(Coop_ROE_Model$coefficients[
  "frac_own_gen"
])
summary(Coop_ROE_Model)

aggregate_coop_data <- coop_data %>%
  subset(!is.na(coop_roe) & !is.na(coop_equity_ratio) & !is.na(coop_ror)) %>%
  select(c(coop_equity, coop_debt, coop_roe, coop_equity_ratio, coop_ror)) %>%
  summarize(
    average_coop_roe = sum(coop_roe * coop_equity) / sum(coop_equity),
    average_coop_equity_ratio = sum(coop_equity) / sum(coop_equity + coop_debt),
    average_coop_ror = sum(coop_ror * (coop_equity + coop_debt)) /
      sum(coop_equity + coop_debt)
  )

average_coop_roe <- as.numeric(aggregate_coop_data$average_coop_roe)
average_coop_ror <- as.numeric(aggregate_coop_data$average_coop_ror)
average_coop_equity_ratio <- as.numeric(
  aggregate_coop_data$average_coop_equity_ratio
)

unit_level_data <- unit_level_data_pudl
tech_map <- tech_map_pudl

unit_level_data <- unit_level_data %>%
  left_join(
    ownership_eia860 %>%
      select(c(
        plant_id_eia,
        generator_id,
        owner_utility_id_eia,
        fraction_owned
      )),
    by = c("plant_id_eia", "generator_id")
  ) %>%
  mutate(
    Utility_ID = coalesce(owner_utility_id_eia, utility_id_eia),
    fraction_owned = coalesce(fraction_owned, 1)
  ) %>%
  group_by(plant_id_eia, generator_id) %>%
  mutate(
    fraction_owned_total = sum(fraction_owned, na.rm = TRUE),
    fraction_owned = if_else(fraction_owned_total == 0, 1, fraction_owned),
    fraction_owned = case_when(
      fraction_owned_total != 1 & fraction_owned_total > 0 ~
        fraction_owned / fraction_owned_total,
      TRUE ~ fraction_owned
    ),
    fraction_owned_total = sum(fraction_owned, na.rm = TRUE),
    fraction_owned = case_when(
      fraction_owned_total != 1 & fraction_owned_total > 0 ~
        fraction_owned / fraction_owned_total,
      TRUE ~ fraction_owned
    ),
    owned_capacity = capacity * fraction_owned
  ) %>%
  ungroup()

unit_level_data <- unit_level_data %>%
  left_join(
    utilities_eia860 %>%
      select(c(utility_id_eia, entity_type)),
    by = c("Utility_ID" = "utility_id_eia")
  )

unit_level_data <- unit_level_data %>%
  left_join(
    utilities_eia860 %>%
      select(c(utility_id_eia, entity_type_UT = entity_type)),
    by = c("utility_id_eia" = "utility_id_eia")
  )

unit_level_data <- unit_level_data %>%
  left_join(
    utility_data_misc_eia861 %>% select(c(utility_id_eia, entity_type_861)),
    by = c("Utility_ID" = "utility_id_eia")
  )

unit_level_data <- unit_level_data %>%
  left_join(
    utilities_eia860 %>%
      select(c(utility_id_eia, entity_type_TD = entity_type)),
    by = c("td_utility_id_eia" = "utility_id_eia")
  )

unit_level_data <- unit_level_data %>%
  left_join(
    utility_data_misc_eia861 %>%
      select(c(utility_id_eia, entity_type_TD_861 = entity_type_861)),
    by = c("td_utility_id_eia" = "utility_id_eia")
  )

unit_level_data <- unit_level_data %>%
  left_join(
    Sector_Entity_Table %>% rename("entity_type_SEC" = "entity_type"),
    by = c("sector_name")
  ) %>%
  mutate(
    entity_type = coalesce(
      entity_type,
      entity_type_861,
      entity_type_UT,
      entity_type_SEC,
      "Q"
    ),
    entity_type_TD = coalesce(
      entity_type_TD,
      entity_type_TD_861,
      entity_type,
      "Q"
    )
  ) %>%
  group_by(Utility_ID, entity_type) %>%
  mutate(n_gens = n_distinct(plant_id_eia, generator_id)) %>%
  ungroup() %>%
  group_by(Utility_ID) %>%
  mutate(
    entity_type = as.factor(if_else(
      n_gens < max(n_gens),
      NA,
      as.character(entity_type)
    )),
    entity_type = as.factor(max(as.character(entity_type), na.rm = TRUE))
  ) %>%
  ungroup() %>%
  select(-c(n_gens)) %>%
  group_by(td_utility_id_eia, entity_type_TD) %>%
  mutate(n_gens = n_distinct(plant_id_eia, generator_id)) %>%
  ungroup() %>%
  group_by(td_utility_id_eia) %>%
  mutate(
    entity_type_TD = as.factor(if_else(
      n_gens < max(n_gens),
      NA,
      as.character(entity_type_TD)
    )),
    entity_type_TD = as.factor(max(as.character(entity_type_TD), na.rm = TRUE))
  ) %>%
  ungroup() %>%
  select(-c(n_gens))

trans_financial_data <- unit_level_data

unit_level_data <- unit_level_data %>%
  left_join(
    operational_data_eia861 %>%
      select(c(
        utility_id_eia,
        total_revenue,
        frac_sales_for_resale_mwh,
        frac_own_gen
      )),
    by = c("Utility_ID" = "utility_id_eia")
  ) %>%
  left_join(
    utility_data_misc_eia861 %>%
      select(c(
        utility_id_eia,
        generation_activity,
        transmission_activity,
        distribution_activity
      )),
    by = c("Utility_ID" = "utility_id_eia")
  ) %>%
  left_join(sales_eia861_summary, by = c("Utility_ID" = "utility_id_eia")) %>%
  left_join(
    state_mapping %>%
      select(c("State", "State_Tax_Rate")),
    by = c("State")
  ) %>%
  left_join(Direct_Pay_Table, by = c("entity_type")) %>%
  left_join(Transferability_Table, by = c("entity_type")) %>%
  group_by(Utility_ID) %>%
  mutate(
    total_revenue = total_revenue,
    State_Tax_Rate = coalesce(
      Average_State_Tax_Rate,
      sum(State_Tax_Rate * owned_capacity, na.rm = TRUE) /
        sum(owned_capacity, na.rm = TRUE),
      sum(State_Tax_Rate, na.rm = TRUE) /
        sum(!is.na(State_Tax_Rate), na.rm = TRUE)
    ),
    Blended_Tax_Rate = case_when(
      entity_type == "I" |
        entity_type == "Q" |
        entity_type == "IND" |
        entity_type == "COM" ~
        Federal_Tax_Rate + State_Tax_Rate * (1 - Federal_Tax_Rate),
      TRUE ~ 0
    ),
    PF_Tax_Rate = Federal_Tax_Rate + State_Tax_Rate * (1 - Federal_Tax_Rate)
  ) %>%
  ungroup() %>%
  left_join(
    coop_data %>%
      select(c(
        Utility_ID,
        coop_roe,
        coop_equity_ratio,
        coop_ror,
        coop_equity,
        coop_debt
      )),
    by = c("Utility_ID")
  ) %>%
  mutate(
    coop_roe = coalesce(
      coop_roe,
      if_else(
        is.na(frac_own_gen),
        average_coop_roe,
        base_coop_roe + coeff_own_gen_coop_roe * frac_own_gen
      )
    ),
    coop_ror = coalesce(
      coop_ror,
      if_else(
        is.na(frac_own_gen),
        average_coop_ror,
        base_coop_ror + coeff_own_gen_coop_ror * frac_own_gen
      )
    ),
    coop_equity_ratio = coalesce(
      coop_equity_ratio,
      if_else(
        is.na(frac_own_gen),
        average_coop_equity_ratio,
        base_coop_er +
          coeff_own_gen_coop_er * frac_own_gen +
          coeff_gen_coop_er * generation_activity +
          coeff_sales_coop_er * frac_sales_for_resale_mwh
      )
    )
  )

## Import and transform Hub data to integrate utility financial data for calculation of revenue requirements implications
## of new asset deployment as well as existing asset securitization
files <- c(
  "net_plant_balance",
  "debt_equity_returns",
  "revenue_by_tech",
  "operations_emissions_by_tech"
)
for (file in files) {
  filename <- paste(cache_dir, "/", file, ".parquet", sep = "")
  if (!file.exists(filename)) {
    arrow::write_parquet(
      read.csv(
        paste(
          "https://utilitytransitionhub.rmi.org/static/data_download/",
          file,
          ".csv",
          sep = ""
        ),
        stringsAsFactors = TRUE
      ),
      filename
    )
  }
  assign(
    paste("hub", file, sep = "_"),
    arrow::read_parquet(filename)
  )
}
aggregate_hub_revenue_by_tech <- aggregate(
  . ~ respondent_id + year + technology + component,
  data = (hub_revenue_by_tech %>%
    filter(detail != "transmission of electricity by others") %>%
    select(-c(parent_name, utility_name, detail, revenue_residential))),
  sum
) %>%
  pivot_wider(names_from = component, values_from = revenue_total) %>%
  filter(!(technology %in% c("adjustment", "purchased_power")))

hub_cap_gen <- hub_operations_emissions_by_tech %>%
  group_by(year, respondent_id, technology_rmi) %>%
  summarize(
    capacity = 1000000 * sum(capacity, na.rm = TRUE),
    net_generation = 1000000 * sum(net_generation, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    FERC_class = case_when(
      technology_rmi == "Steam" ~ "steam",
      technology_rmi == "Hydro" ~ "hydro",
      technology_rmi == "Nuclear" ~ "nuclear",
      technology_rmi == "Other Fossil" ~ "other_fossil",
      technology_rmi == "Renewables & Storage" ~ "renewables",
      TRUE ~ NA
    ),
    FERC_class = as.factor(FERC_class)
  )

hub_data <- hub_net_plant_balance %>%
  filter(
    !(FERC_class %in%
      c("general", "intangible", "regional_transmission_and_market_operation"))
  ) %>%
  left_join(
    aggregate_hub_revenue_by_tech,
    by = c(
      "respondent_id" = "respondent_id",
      "year" = "year",
      "FERC_class" = "technology"
    )
  ) %>%
  left_join(
    hub_cap_gen %>% filter(!is.na(FERC_class)) %>% select(-c(technology_rmi)),
    by = c("respondent_id", "year" = "year", "FERC_class")
  ) %>%
  group_by(respondent_id) %>%
  mutate(
    frac_depr = if_else(original_cost != 0, accum_depr / original_cost, NA),
    frac_depr = if_else(frac_depr > 1 | frac_depr < 0, NA, frac_depr),
    dep_amort_exp = (if_else(
      is.na(depreciation_expense) | depreciation_expense < 0,
      0,
      depreciation_expense
    ) +
      if_else(
        is.na(`amort_&_depl_of_utility_plant`) |
          `amort_&_depl_of_utility_plant` < 0,
        0,
        `amort_&_depl_of_utility_plant`
      )),
    depreciation_rate = dep_amort_exp / original_cost,
    OM_exp = (if_else(
      is.na(non_fuel_operation_expenses) | non_fuel_operation_expenses < 0,
      0,
      non_fuel_operation_expenses
    ) +
      if_else(
        is.na(maintenance_expenses) | maintenance_expenses < 0,
        0,
        maintenance_expenses
      )),
    capex_per_kw_FERC_tech = if_else(
      !is.na(capacity) & capacity > 0,
      original_cost / capacity,
      NA
    ),
    opex_per_kw_FERC_tech = if_else(
      !is.na(capacity) & capacity > 0,
      OM_exp / capacity,
      NA
    ),
    fixed_OM_frac = OM_exp / original_cost,
    # fixed_OM_frac = if_else(FERC_class=="transmission" & fixed_OM_frac > 0.06,NA,fixed_OM_frac),
    # depreciation_rate = if_else(depreciation_rate>0.1 | depreciation_rate<0.01,NA,depreciation_rate),
    rem_life = if_else(
      depreciation_rate > 0,
      (1 - frac_depr) / depreciation_rate,
      NA
    )
  ) %>%
  ungroup()

reg_lag <- 3

clean_hub_data <- hub_data %>%
  select(c(
    respondent_id,
    FERC_class,
    original_cost,
    capacity,
    net_generation,
    dep_amort_exp,
    year,
    OM_exp
  )) %>%
  group_by(respondent_id, FERC_class) %>%
  filter(original_cost > 0) %>%
  arrange(year) %>%
  mutate(
    delta_capex = if_else(
      year >= min(year) + reg_lag,
      original_cost - lag(original_cost, reg_lag),
      NA
    ) /
      original_cost,
    delta_capacity = if_else(
      capacity > 0 & year >= min(year) + reg_lag,
      1 - lag(capacity, reg_lag) / capacity,
      NA
    ),
    delta_gen = if_else(
      net_generation > 0 & year >= min(year) + reg_lag,
      1 - lag(net_generation, reg_lag) / net_generation,
      NA
    ),
    delta_OM_exp = if_else(
      year >= min(year) + reg_lag,
      OM_exp - lag(OM_exp, reg_lag),
      NA
    ) /
      original_cost,
    delta_dep_exp = if_else(
      year >= min(year) + reg_lag,
      dep_amort_exp - lag(dep_amort_exp, reg_lag),
      NA
    ) /
      original_cost,
  ) %>%
  ungroup() %>%
  filter(
    !(is.na(delta_capex) |
      is.na(delta_OM_exp) |
      OM_exp == 0 |
      original_cost == 0)
  )

# average_transmission_fixed_OM_frac <- as.numeric(coef(lm(delta_OM_exp ~ 0 + delta_capex,
#                                                         data = clean_hub_data %>% filter(FERC_class=="transmission")))[1])
# average_transmission_depreciation_rate <- as.numeric(coef(lm(delta_dep_exp ~ 0 + delta_capex,
#                                                             data = clean_hub_data %>% filter(FERC_class=="transmission")))[1])

# ave_OM_frac_model <- lm(delta_OM_exp ~ 0 + delta_capex,data = clean_hub_data %>% filter(FERC_class=="transmission"))
# summary(ave_OM_frac_model)

# ave_dep_frac_model <- lm(delta_dep_exp ~ 0 + delta_capex,data = clean_hub_data %>% filter(FERC_class=="transmission"))
# summary(ave_dep_frac_model)

aggregate_OM_frac_data <- clean_hub_data %>%
  group_by(respondent_id, FERC_class) %>%
  do(tidy(lm(delta_OM_exp ~ 0 + delta_capex, data = .))) %>%
  select(c(
    respondent_id,
    Technology_FERC = FERC_class,
    fixed_OM_frac_regr = estimate,
    p.value
  )) %>%
  ungroup() %>%
  mutate(
    fixed_OM_frac_regr = if_else(
      !is.na(p.value) & p.value < 0.05 & fixed_OM_frac_regr > 0,
      fixed_OM_frac_regr,
      NA
    )
  ) %>%
  select(-c(p.value))

aggregate_dep_rate_data <- clean_hub_data %>%
  group_by(respondent_id, FERC_class) %>%
  do(tidy(lm(delta_dep_exp ~ 0 + delta_capex, data = .))) %>%
  select(c(
    respondent_id,
    Technology_FERC = FERC_class,
    depreciation_rate_regr = estimate,
    p.value
  )) %>%
  ungroup() %>%
  mutate(
    depreciation_rate_regr = if_else(
      !is.na(p.value) & p.value < 0.05 & depreciation_rate_regr > 0,
      depreciation_rate_regr,
      NA
    )
  ) %>%
  select(-c(p.value))

aggregate_hub_data <- hub_data %>%
  select(c(
    year,
    FERC_class,
    original_cost,
    capacity,
    accum_depr,
    net_plant_balance,
    `amort_&_depl_of_utility_plant`,
    depreciation_expense,
    maintenance_expenses,
    fixed_OM_frac,
    non_fuel_operation_expenses
  )) %>%
  subset(original_cost > 0 & accum_depr > 0) %>%
  group_by(FERC_class, year) %>%
  summarize(
    average_frac_depr = sum(accum_depr) / sum(original_cost),
    average_capex_per_kw_FERC = if_else(
      sum(capacity, na.rm = TRUE) > 0,
      sum(original_cost) / sum(capacity, na.rm = TRUE),
      NA
    ),
    average_depreciation_rate = (sum(
      if_else(depreciation_expense > 0, depreciation_expense, 0),
      na.rm = TRUE
    ) +
      sum(
        if_else(
          `amort_&_depl_of_utility_plant` > 0,
          `amort_&_depl_of_utility_plant`,
          0
        ),
        na.rm = TRUE
      )) /
      sum(
        if_else(
          ((is.na(depreciation_expense) | depreciation_expense < 0) &
            (is.na(`amort_&_depl_of_utility_plant`) |
              `amort_&_depl_of_utility_plant` < 0)),
          NA,
          original_cost
        ),
        na.rm = TRUE
      ),
    average_fixed_OM_frac = (sum(
      if_else(
        non_fuel_operation_expenses > 0 & !is.na(fixed_OM_frac),
        non_fuel_operation_expenses,
        0
      ),
      na.rm = TRUE
    ) +
      sum(
        if_else(
          maintenance_expenses > 0 & !is.na(fixed_OM_frac),
          maintenance_expenses,
          0
        ),
        na.rm = TRUE
      )) /
      sum(
        if_else(
          ((is.na(non_fuel_operation_expenses) |
            non_fuel_operation_expenses < 0) &
            (is.na(maintenance_expenses) | maintenance_expenses < 0)) |
            is.na(fixed_OM_frac),
          NA,
          original_cost
        ),
        na.rm = TRUE
      ),
    average_rem_life = (1 - average_frac_depr) / average_depreciation_rate
  ) %>%
  ungroup()

# Find net plant balance by utility
hub_net_plant_balance_2020 <- hub_net_plant_balance %>%
  select(c(
    respondent_id,
    year,
    Technology_FERC = FERC_class,
    net_plant_balance
  )) %>%
  filter(year == 2020) %>%
  group_by(respondent_id, Technology_FERC) %>%
  summarise(utility_net_plant_balance = sum(net_plant_balance)) %>%
  ungroup()

# Read in FERC Rate Case Data
ferc_SP_utility_map <- readxl::read_excel(
  paste(
    cloud$AZURE_CACHE_PATH,
    cloud$cached_path(
      "patio-restricted/ferc_SP_utility_map.xlsm",
      download = TRUE
    ),
    sep = "/"
  ),
  sheet = "ferc_SP_utility_map"
) %>%
  arrange(`FERC Utility ID`) %>%
  drop_na("FERC Utility ID") %>%
  select(c(
    "respondent_id" = "FERC Utility ID",
    "ror" = "Most Recent ROR",
    "roe" = "Most Recent ROE",
    "equity_ratio" = "Most Recent Equity Ratio",
    "rating" = "Fuzzy S&P Credit Rating"
  )) %>%
  filter(ror > 0) %>%
  left_join(
    hub_net_plant_balance_2020 %>%
      group_by(respondent_id) %>%
      summarize(
        utility_net_plant_balance = sum(utility_net_plant_balance, na.rm = TRUE)
      ) %>%
      ungroup(),
    by = "respondent_id"
  ) %>%
  mutate(
    implied_debt_cost = (ror - (equity_ratio * roe)) / (1 - equity_ratio)
  ) %>%
  arrange(implied_debt_cost) %>%
  mutate(
    debt_cost_perc = cumsum(if_else(
      implied_debt_cost == 0 | is.na(implied_debt_cost),
      1,
      if_else(is.na(utility_net_plant_balance), 0, utility_net_plant_balance)
    )) /
      sum(utility_net_plant_balance, na.rm = TRUE),
    debt_spread = gsub("[+-]", "", rating),
    debt_spread = coalesce(
      debt_spread,
      as.factor(case_when(
        # debt_cost_perc <= 1/24 ~ "AAA",
        # debt_cost_perc <= 1/12 ~ "AA",
        debt_cost_perc <= 1 / 2 ~ "A",
        TRUE ~ "BBB"
      ))
    )
  ) %>%
  select(-c(utility_net_plant_balance, debt_cost_perc, implied_debt_cost))

aggregate_ferc_SP <- ferc_SP_utility_map %>%
  left_join(hub_net_plant_balance_2020, by = "respondent_id") %>%
  group_by(Technology_FERC) %>%
  summarize(
    average_iou_ror = sum(
      if_else(is.na(ror), 0, ror) * utility_net_plant_balance,
      na.rm = TRUE
    ) /
      sum(if_else(is.na(ror), 0, utility_net_plant_balance), na.rm = TRUE),
    average_iou_roe = sum(
      if_else(is.na(roe), 0, roe) * utility_net_plant_balance,
      na.rm = TRUE
    ) /
      sum(if_else(is.na(roe), 0, utility_net_plant_balance), na.rm = TRUE),
    average_iou_equity_ratio = sum(
      if_else(is.na(equity_ratio), 0, equity_ratio) * utility_net_plant_balance,
      na.rm = TRUE
    ) /
      sum(
        if_else(is.na(equity_ratio), 0, utility_net_plant_balance),
        na.rm = TRUE
      ),
    average_iou_debt_spread = "BBB"
  ) %>%
  ungroup()

Technology_List <- hub_data %>%
  select(c(Technology_FERC = FERC_class)) %>%
  distinct()

gen_financial_data <- unit_level_data %>%
  select(c(
    Utility_ID,
    entity_type,
    Direct_Pay,
    Transferability,
    cust_revenue,
    cust_sales_mwh,
    Average_State_Tax_Rate,
    com_frac_revenue,
    ind_frac_revenue,
    other_frac_revenue,
    res_frac_revenue,
    trans_frac_revenue,
    com_frac_sales_mwh,
    ind_frac_sales_mwh,
    other_frac_sales_mwh,
    res_frac_sales_mwh,
    trans_frac_sales_mwh,
    generation_activity,
    transmission_activity,
    distribution_activity,
    total_revenue,
    frac_sales_for_resale_mwh,
    frac_own_gen,
    State_Tax_Rate,
    Blended_Tax_Rate,
    PF_Tax_Rate,
    coop_roe,
    coop_equity_ratio,
    coop_ror,
    coop_equity,
    coop_debt
  )) %>%
  distinct() %>%
  cross_join(Technology_List, copy = TRUE) %>%
  filter(Technology_FERC != "transmission")

trans_financial_data <- trans_financial_data %>%
  left_join(
    operational_data_eia861 %>%
      select(c(
        utility_id_eia,
        total_revenue,
        frac_sales_for_resale_mwh,
        frac_own_gen
      )),
    by = c("td_utility_id_eia" = "utility_id_eia")
  ) %>%
  left_join(
    utility_data_misc_eia861 %>%
      select(c(
        utility_id_eia,
        generation_activity,
        transmission_activity,
        distribution_activity
      )),
    by = c("td_utility_id_eia" = "utility_id_eia")
  ) %>%
  left_join(
    sales_eia861_summary,
    by = c("td_utility_id_eia" = "utility_id_eia")
  ) %>%
  left_join(
    state_mapping %>%
      select(c("State", "State_Tax_Rate")),
    by = c("State")
  ) %>%
  left_join(Direct_Pay_Table, by = c("entity_type_TD" = "entity_type")) %>%
  left_join(Transferability_Table, by = c("entity_type_TD" = "entity_type")) %>%
  group_by(td_utility_id_eia) %>%
  mutate(
    total_revenue = total_revenue,
    State_Tax_Rate = coalesce(
      Average_State_Tax_Rate,
      sum(State_Tax_Rate * owned_capacity, na.rm = TRUE) /
        sum(owned_capacity, na.rm = TRUE),
      sum(State_Tax_Rate, na.rm = TRUE) /
        sum(!is.na(State_Tax_Rate), na.rm = TRUE)
    ),
    Blended_Tax_Rate = case_when(
      entity_type_TD == "I" |
        entity_type_TD == "Q" |
        entity_type_TD == "IND" |
        entity_type_TD == "COM" ~
        Federal_Tax_Rate + State_Tax_Rate * (1 - Federal_Tax_Rate),
      TRUE ~ 0
    ),
    PF_Tax_Rate = Federal_Tax_Rate + State_Tax_Rate * (1 - Federal_Tax_Rate)
  ) %>%
  ungroup() %>%
  left_join(
    coop_data %>%
      select(c(
        Utility_ID,
        coop_roe,
        coop_equity_ratio,
        coop_ror,
        coop_equity,
        coop_debt
      )),
    by = c("td_utility_id_eia" = "Utility_ID")
  ) %>%
  mutate(
    coop_roe = coalesce(
      coop_roe,
      if_else(
        is.na(frac_own_gen),
        average_coop_roe,
        base_coop_roe + coeff_own_gen_coop_roe * frac_own_gen
      )
    ),
    coop_ror = coalesce(
      coop_ror,
      if_else(
        is.na(frac_own_gen),
        average_coop_ror,
        base_coop_ror + coeff_own_gen_coop_ror * frac_own_gen
      )
    ),
    coop_equity_ratio = coalesce(
      coop_equity_ratio,
      if_else(
        is.na(frac_own_gen),
        average_coop_equity_ratio,
        base_coop_er +
          coeff_own_gen_coop_er * frac_own_gen +
          coeff_gen_coop_er * generation_activity +
          coeff_sales_coop_er * frac_sales_for_resale_mwh
      )
    ),
    Technology_FERC = "transmission"
  ) %>%
  select(c(
    Utility_ID = td_utility_id_eia,
    entity_type = entity_type_TD,
    Technology_FERC,
    Direct_Pay,
    Transferability,
    cust_revenue,
    cust_sales_mwh,
    Average_State_Tax_Rate,
    com_frac_revenue,
    ind_frac_revenue,
    other_frac_revenue,
    res_frac_revenue,
    trans_frac_revenue,
    com_frac_sales_mwh,
    ind_frac_sales_mwh,
    other_frac_sales_mwh,
    res_frac_sales_mwh,
    trans_frac_sales_mwh,
    generation_activity,
    transmission_activity,
    distribution_activity,
    total_revenue,
    frac_sales_for_resale_mwh,
    frac_own_gen,
    State_Tax_Rate,
    Blended_Tax_Rate,
    PF_Tax_Rate,
    coop_roe,
    coop_equity_ratio,
    coop_ror,
    coop_equity,
    coop_debt
  )) %>%
  distinct()

utility_financial_data <- rbind(gen_financial_data, trans_financial_data)

full_parquet_utilities <- full_parquet_map %>%
  select(c(Utility_ID = utility_id_eia)) %>%
  distinct()
generation_owners <- unit_level_data %>%
  select(c(Utility_ID)) %>%
  distinct()
transmission_owners <- transmission_owners_details %>%
  select(c(Utility_ID = td_utility_id_eia)) %>%
  distinct()
# other_utilities <- Utilities %>% select(c(Utility_ID = utility_id_eia)) %>% distinct()

asset_owners <- rbind(
  full_parquet_utilities,
  generation_owners,
  transmission_owners
) %>%
  distinct() %>%
  left_join(
    utilities_inputs %>% select(c(utility_id_eia, respondent_id)),
    by = c("Utility_ID" = "utility_id_eia"),
    relationship = "many-to-many"
  )

## Transmission costs

## Import cleaned transmission line cost and characteristics data from FERC Form 1, matched to estimate transmission line
## parameters via a python module. Then estimate Surge Impedence Loading (SIL) based on line parameters. Finally, based on
## brutally averaging a St. Clair curve, approximate line capacity as 1.5 x SIL
voltage_spacing <- cloud$read_cloud_file(
  "patio-data/20241031/voltage_spacing_3phase.parquet"
)
transmission_capex <- cloud$read_cloud_file(
  "patio-data/20241031/f1_transmission_cleaned.parquet"
) %>%
  filter(
    row_literal != "TOTAL" &
      cost_land + cost_poles + cost_cndctr > 1000000 &
      crct_present > 0 &
      line_length * crct_present >= 10 &
      voltage > 0
  ) %>%
  left_join(CPIU, by = c("year" = "Year")) %>%
  left_join(voltage_spacing, by = c("voltage" = "voltage")) %>%
  mutate(
    voltage = if_else(voltage > 1000, voltage / 1000, voltage),
    cable_radius_ft = diameter_inch_comp_cable_OD / 24,
    resistance_per_mile = resistance_ohms_kft_AC_75C * 5.28,
    bundles = if_else(
      pmax(
        1,
        pmin(
          cnd_size_1,
          cnd_size_2,
          cnd_size_3,
          cnd_size_4,
          cnd_size_5,
          na.rm = TRUE
        ),
        na.rm = TRUE
      ) >=
        8,
      1,
      pmax(
        1,
        pmin(
          cnd_size_1,
          cnd_size_2,
          cnd_size_3,
          cnd_size_4,
          cnd_size_5,
          na.rm = TRUE
        ),
        na.rm = TRUE
      )
    ),
    GMR_ft = cable_radius_ft * 0.7788, # Geometric mean radius of the composite cable accounting for inductance within cable
    GMR_bundle = (bundles *
      GMR_ft *
      if_else(bundles > 1, (1.5 / (2 * sinpi(1 / bundles)))^(bundles - 1), 1)),
    r_bundle = (bundles *
      cable_radius_ft *
      if_else(bundles > 1, (1.5 / (2 * sinpi(1 / bundles)))^(bundles - 1), 1)),
    X_L = 2.022 * 10^(-3) * 60 * log(spacing_ft / GMR_bundle),
    X_C = 1.779 * 10^6 * (1 / 60) * log(spacing_ft / cable_radius_ft),
    Z_0 = (X_L * X_C)^(1 / 2),
    SIL = crct_present * voltage^2 / Z_0,
    line_capacity_thermal_limit = crct_present *
      3^(1 / 2) *
      ampacity *
      voltage /
      1000,
    line_capacity = 1.5 * SIL
  )

## Estimate transmission line costs per mile x kW line capacity based on FERC costs and line capacity estimates
transmission_capex <- transmission_capex[, c(
  "respondent_id",
  "year",
  "line_length",
  "crct_present",
  "voltage",
  "line_capacity",
  "cost_land",
  # "cnd_size",
  "cost_poles",
  "cost_cndctr",
  "asset_retire_cost",
  "Inflation_Factor_2021",
  "cost_total"
)] %>%
  mutate(
    # line_crct = line_length * crct_present,
    # line_crct_size = line_crct * cnd_size,
    cost_build = (cost_land + cost_poles + cost_cndctr) / Inflation_Factor_2021
  )

clean_trans_data <- transmission_capex %>%
  filter(line_length > 0 & line_capacity > 0 & cost_build > 0) %>%
  select(c(respondent_id, line_length, line_capacity, cost_build)) %>%
  mutate(line_capacity_x_length = line_length * line_capacity)


average_transmission_CAPEX <- as.numeric(coef(lm(
  cost_build ~ 0 + line_capacity_x_length,
  data = clean_trans_data
))[1]) /
  1000
# trans_line_cost_model2 <- lm(cost_build ~ line_capacity_x_length,data = clean_trans_data)
# anova(trans_line_cost_model2,trans_line_cost_model)

aggregate_trans_data <- clean_trans_data %>%
  group_by(respondent_id) %>%
  do(tidy(lm(cost_build ~ 0 + line_capacity_x_length, data = .))) %>%
  select(c(
    respondent_id,
    Transmission_CAPEX = estimate,
    p.value
  )) %>%
  ungroup() %>%
  mutate(
    Transmission_CAPEX = if_else(
      !is.na(p.value) & p.value < 0.05,
      Transmission_CAPEX / 1000,
      average_transmission_CAPEX
    )
  )

asset_owners <- asset_owners %>%
  cross_join(Technology_List, copy = TRUE) %>%
  left_join(
    hub_data %>%
      filter(year == 2020) %>%
      select(c(
        respondent_id,
        Technology_FERC = FERC_class,
        original_cost,
        capacity,
        accum_depr,
        net_plant_balance,
        returns,
        depreciation_expense,
        `amort_&_depl_of_utility_plant`,
        non_fuel_operation_expenses,
        maintenance_expenses
      )),
    by = c("respondent_id", "Technology_FERC"),
    relationship = "many-to-many"
  ) %>%
  left_join(ferc_SP_utility_map, by = c("respondent_id")) %>%
  left_join(
    aggregate_trans_data %>% select(c(respondent_id, Transmission_CAPEX)),
    by = "respondent_id"
  ) %>%
  left_join(
    aggregate_OM_frac_data,
    by = c("respondent_id", "Technology_FERC")
  ) %>%
  left_join(aggregate_dep_rate_data, by = c("respondent_id", "Technology_FERC"))

# Merge net plant balances, returns, depreciation expenses

asset_owners <- asset_owners %>%
  mutate(
    include_in_calcs = (original_cost > 0) &
      (accum_depr > 0) &
      (!((is.na(depreciation_expense) | depreciation_expense < 0) &
        (is.na(`amort_&_depl_of_utility_plant`) |
          `amort_&_depl_of_utility_plant`))) &
      (!((is.na(non_fuel_operation_expenses) |
        non_fuel_operation_expenses < 0) &
        (is.na(maintenance_expenses) | maintenance_expenses < 0)))
  ) %>%
  group_by(Utility_ID, Technology_FERC) %>%
  summarize(
    Transmission_CAPEX = sum(
      if_else(
        is.na(Transmission_CAPEX) | Technology_FERC != "transmission",
        0,
        Transmission_CAPEX
      ) *
        net_plant_balance
    ) /
      sum(if_else(
        is.na(Transmission_CAPEX) | Technology_FERC != "transmission",
        0,
        net_plant_balance
      )),
    ror = sum(if_else(is.na(ror), 0, ror) * net_plant_balance) /
      sum(if_else(is.na(ror), 0, net_plant_balance)),
    roe = sum(if_else(is.na(roe), 0, roe) * net_plant_balance) /
      sum(if_else(is.na(roe), 0, net_plant_balance)),
    debt_spread = as.factor(case_when(
      n_distinct(debt_spread) > 1 ~ "A",
      n_distinct(debt_spread) == 1 ~
        min(as.character(debt_spread), na_rm = TRUE),
      TRUE ~ NA
    )),
    equity_ratio = sum(
      if_else(is.na(equity_ratio), 0, equity_ratio) * net_plant_balance
    ) /
      sum(if_else(is.na(equity_ratio), 0, net_plant_balance)),
    original_cost = sum(
      if_else(include_in_calcs, original_cost, 0),
      na.rm = TRUE
    ),
    capacity = sum(if_else(include_in_calcs, capacity, 0)),
    accum_depr = sum(if_else(include_in_calcs, accum_depr, 0), na.rm = TRUE),
    net_plant_balance = sum(
      if_else(include_in_calcs, net_plant_balance, 0),
      na.rm = TRUE
    ),
    returns = sum(if_else(include_in_calcs, returns, 0), na.rm = TRUE),
    depreciation_expense = sum(
      if_else(include_in_calcs, depreciation_expense, 0),
      na.rm = TRUE
    ),
    `amort_&_depl_of_utility_plant` = sum(
      if_else(include_in_calcs, `amort_&_depl_of_utility_plant`, 0),
      na.rm = TRUE
    ),
    non_fuel_operation_expenses = sum(
      if_else(include_in_calcs, non_fuel_operation_expenses, 0),
      na.rm = TRUE
    ),
    maintenance_expenses = sum(
      if_else(include_in_calcs, maintenance_expenses, 0),
      na.rm = TRUE
    )
  ) %>%
  ungroup() %>%
  mutate(
    frac_depr = if_else(original_cost != 0, accum_depr / original_cost, NA),
    capex_per_kw_FERC = if_else(capacity != 0, original_cost / capacity, NA),
    depreciation_rate = (if_else(
      is.na(depreciation_expense) | depreciation_expense < 0,
      0,
      depreciation_expense
    ) +
      if_else(
        is.na(`amort_&_depl_of_utility_plant`) |
          `amort_&_depl_of_utility_plant` < 0,
        0,
        `amort_&_depl_of_utility_plant`
      )) /
      original_cost,
    fixed_OM_frac = (if_else(
      is.na(non_fuel_operation_expenses) | non_fuel_operation_expenses < 0,
      0,
      non_fuel_operation_expenses
    ) +
      if_else(
        is.na(maintenance_expenses) | maintenance_expenses < 0,
        0,
        maintenance_expenses
      )) /
      original_cost,
    rem_life = if_else(
      depreciation_rate > 0,
      (1 - frac_depr) / depreciation_rate,
      NA
    )
  ) %>%
  left_join(aggregate_ferc_SP, by = "Technology_FERC") %>%
  left_join(
    aggregate_hub_data %>% filter(year == 2020) %>% select(-c("year")),
    by = c("Technology_FERC" = "FERC_class")
  ) %>%
  left_join(utility_financial_data, by = c("Utility_ID", "Technology_FERC")) %>%
  filter(!is.na(entity_type)) %>%
  mutate(
    coop_roe = coalesce(coop_roe, average_coop_roe),
    coop_ror = coalesce(coop_ror, average_coop_ror),
    coop_equity_ratio = coalesce(coop_equity_ratio, average_coop_equity_ratio),
    Transmission_CAPEX = if_else(
      Technology_FERC != "transmission",
      NA,
      coalesce(Transmission_CAPEX, average_transmission_CAPEX)
    ),
    frac_depr = coalesce(frac_depr, average_frac_depr),
    capex_per_kw_FERC = coalesce(capex_per_kw_FERC, average_capex_per_kw_FERC),
    depreciation_rate = coalesce(
      if_else(
        depreciation_rate < 0.01,
        average_depreciation_rate,
        depreciation_rate
      ),
      average_depreciation_rate
    ),
    fixed_OM_frac = coalesce(
      if_else(
        Technology_FERC == "transmission" & fixed_OM_frac > 0.06,
        average_fixed_OM_frac,
        fixed_OM_frac
      ),
      average_fixed_OM_frac
    ),
    rem_life = coalesce(rem_life, average_rem_life),
    ror = coalesce(ror, if_else(entity_type == "C", coop_ror, average_iou_ror)),
    roe = coalesce(roe, if_else(entity_type == "C", coop_roe, average_iou_roe)),
    equity_ratio = coalesce(
      equity_ratio,
      if_else(entity_type == "C", coop_equity_ratio, average_iou_equity_ratio)
    ),
    debt_spread = as.factor(coalesce(debt_spread, average_iou_debt_spread)),
    debt_cost = (ror - (equity_ratio * roe)) / (1 - equity_ratio),
    adjusted_ror = (roe * equity_ratio) +
      ((debt_cost * (1 - equity_ratio)) * (1 - Blended_Tax_Rate))
  )

# Merge in ror, roe, equity ratio, debt cost, adjusted ror by tech
unit_level_data <- unit_level_data %>%
  left_join(
    asset_owners %>%
      select(c(
        Utility_ID,
        Technology_FERC,
        frac_depr,
        depreciation_rate,
        fixed_OM_frac,
        rem_life,
        ror,
        roe,
        equity_ratio,
        debt_cost,
        adjusted_ror
      )),
    by = c("Utility_ID", "Technology_FERC")
  ) %>%
  mutate(
    coop_equity_ratio = coalesce(coop_equity_ratio, average_coop_equity_ratio)
  )

print("Writing unit financial inputs")
arrow::write_parquet(
  unit_level_data,
  paste(
    root_dir,
    "/econ_results/",
    patio_results,
    "_unit_financial_inputs.parquet",
    sep = ""
  )
)
arrow::write_parquet(
  asset_owners,
  paste(
    root_dir,
    "/econ_results/",
    patio_results,
    "_asset_owners.parquet",
    sep = ""
  )
)

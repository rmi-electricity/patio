
#Load the RCurl library to give us HTTP/FTP functions
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
#library(igraph)

#setwd("~/My Drive/GitHub/patio-model/r_data")
if (!grepl("r_data",getwd())) setwd(paste(getwd(), "r_data", sep="/"))

# We now begin the process of reading in 860, 923, and CEMS data to create a complete historical dataset
# characterizing all EIA reporting generators, and estimating their fuel costs. We will use some of that data
# to better characterize the FERC data (parasitic load to generate estimate gross generation) and estimate
# opex and capex coefficients from that dataset. We then merge the coefficients with the EIA generators to
# develop complete estimates of start-up / stopping costs, fuel costs, as well as variable and fixed O&M
# and capex. We do this both historically, and in a counterfactual dataset for our subsequent economic modeling

# First, read in final fuel costs processed in the "Fuel Cost Calculations" script based on EIA-923 fuel receipts data

final_fuel_costs <- read_parquet("final_fuel_costs.parquet")
#generation_fuel_table <- read_excel("generation_fuel_923_fossil_plants_2008_2020 - 2022-03-14 - values.xlsx")

final_fuel_costs <- final_fuel_costs %>%
  subset(select = c(
    report_date,
    plant_id_eia,
    prime_mover_code,
    year,
    fuel_group_code,
    energy_source_code,
    fuel_consumed_mmbtu_gen_fuel_level,
    final_fuel_cost_per_mmbtu
  )) %>%
  mutate(
    report_month = month(report_date),
    final_fuel_cost = fuel_consumed_mmbtu_gen_fuel_level * final_fuel_cost_per_mmbtu,
    prime_mover_code =
      case_when(
        prime_mover_code == "CA" ~ "CC",
        prime_mover_code == "CT" ~ "CC",
        prime_mover_code == "CS" ~ "CC",
        TRUE ~ prime_mover_code
      )
  ) %>% subset(select = -c(report_date)) %>%
  rename(
    "report_year" = "year"
  ) %>% group_by(report_year,report_month,fuel_group_code,plant_id_eia,prime_mover_code) %>%
  summarize(
    fuel_consumed_mmbtu_gen_fuel_level = sum(ifelse(is.na(fuel_consumed_mmbtu_gen_fuel_level),0,fuel_consumed_mmbtu_gen_fuel_level)),
    final_fuel_cost = sum(ifelse(is.na(final_fuel_cost),0,final_fuel_cost)),
    final_fuel_cost_per_mmbtu = ifelse(!is.na(fuel_consumed_mmbtu_gen_fuel_level) & fuel_consumed_mmbtu_gen_fuel_level>0,
                                       final_fuel_cost / fuel_consumed_mmbtu_gen_fuel_level,
                                       final_fuel_cost_per_mmbtu)
  ) %>% ungroup() %>%
  group_by(report_year,report_month,fuel_group_code,plant_id_eia) %>%
  mutate(
    fuel_consumed_mmbtu_plant = sum(ifelse(is.na(fuel_consumed_mmbtu_gen_fuel_level),0,fuel_consumed_mmbtu_gen_fuel_level)),
    final_fuel_cost_plant = sum(ifelse(is.na(final_fuel_cost),0,final_fuel_cost)),
    final_fuel_cost_per_mmbtu = case_when(
      !is.na(fuel_consumed_mmbtu_gen_fuel_level) & fuel_consumed_mmbtu_gen_fuel_level>0 ~ final_fuel_cost_per_mmbtu,
      fuel_consumed_mmbtu_plant > 0 ~ final_fuel_cost_plant / fuel_consumed_mmbtu_plant,
      TRUE ~ final_fuel_cost_per_mmbtu
    )
  ) %>% select(-c(
    final_fuel_cost_plant,
    fuel_consumed_mmbtu_plant,
    final_fuel_cost,
    fuel_consumed_mmbtu_gen_fuel_level
  )) %>% ungroup()

# Next, we read in the fuel group and emissions map we have generated from EPA data and EIA documentation

fuel_group_and_emissions_map <- read_excel("fuel_group_and_emissions_map.xlsx")

fuel_map <- fuel_group_and_emissions_map %>% select(c(energy_source_code,fuel_group_code))

# Now, pull in current unit level data from EIA-860M and process for use to create both historical generation / fuel use and
# a counterfactual data set for use in the Patio model

all_years <- c(2008:2020)
all_months <- c(1:12)

unit_level_data <- read_parquet("../patio_data/unit_level_costs_with_flag.parquet")

unit_level_data <- unit_level_data %>%
  rename(
    "plant_id_eia" = "Plant_ID",
    "prime_mover_code" = "Prime_with_CCs",
    "generator_id" = "Generator_ID"
  )

# Create plant level grouping file for aggregation

plant_grouping <- unit_level_data %>%
  select(c(
    plant_id_eia,
    State,
    Balancing_Authority_Code,
    Latitude,
    Longitude
  )) %>% distinct()

# Next, we create unit level data for all years and months for each operating and retired asset

unit_level_data <- unit_level_data %>%
  left_join(all_years, by = character(), copy = TRUE) %>% rename(report_year = y) %>%
  left_join(all_months, by = character(), copy = TRUE) %>% rename(report_month = y) %>%
  mutate(month_hours = unname(24*days_in_month(ymd(report_year*10000+report_month*100+1))))

# Filter unit level data for all years and months to create a list of plant and generator ids of
# assets operating in each month and year for historical analysis

units_present_hist <- unit_level_data %>%
  filter(
    ((Operating_Year < report_year) | ((Operating_Year==report_year) & (Operating_Month <= report_month))) &
      (is.na(Retirement_Year) |
         (Retirement_Year > report_year) |
         ((Retirement_Year==report_year ) & (Retirement_Month > report_month))) &
      (is.na(Planned_Retirement_Year) |
         (Planned_Retirement_Year > report_year) |
         ((Planned_Retirement_Year==report_year ) & (Planned_Retirement_Month > report_month)))) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_year,
    report_month,
    month_hours
  ))

# Read in historical monthly generator level net generation data from historical EIA 923 data

gen_monthly_net_mwh <- read_parquet("generators_923_historical.parquet") %>%
  mutate(
    #report_date = report_date + hours(8),
    #report_year = year(report_date),
    #report_month = month(report_date),
    net_generation_mwh = ifelse(is.na(net_generation_mwh),0,net_generation_mwh)
  ) %>%
  select(c(
    report_year,
    report_month,
    plant_id_eia,
    generator_id,
    net_generation_mwh
  ))

# Read in CAMD crosswalk to EIA and generate unique CAMD groups for allocation of CEMS data to EIA generators

CAMD_crosswalk <- read_parquet("camd_eia_crosswalk_by_year.parquet") %>%
  select(c(
    capacity_year,
    CAMD_PLANT_ID,
    CAMD_UNIT_ID,
    CAMD_GENERATOR_ID,
    CAMD_NAMEPLATE_CAPACITY,
    EIA_PLANT_ID,
    EIA_NAMEPLATE_CAPACITY_HIST,
    EIA_NAMEPLATE_CAPACITY,
    EIA_UNIT_TYPE,
    EIA_UNIT_CODE,
    EIA_GENERATOR_ID
  )) %>%
  filter(!is.na(EIA_PLANT_ID)) %>%
  mutate(
    CAMD_line = row_number(),
    EIA_NAMEPLATE_CAPACITY_HIST = ifelse(is.na(EIA_NAMEPLATE_CAPACITY_HIST),EIA_NAMEPLATE_CAPACITY,EIA_NAMEPLATE_CAPACITY_HIST)
  ) %>%
  group_by(EIA_PLANT_ID,EIA_GENERATOR_ID,capacity_year) %>%
  mutate(CAMD_group = max(CAMD_line)) %>% ungroup()

for (iter in 1:20) {
  CAMD_crosswalk <- CAMD_crosswalk %>%
    group_by(CAMD_PLANT_ID,CAMD_UNIT_ID,capacity_year) %>%
    mutate(CAMD_group = max(CAMD_group)) %>% ungroup() %>%
    group_by(EIA_PLANT_ID,EIA_GENERATOR_ID,capacity_year) %>%
    mutate(CAMD_group = max(CAMD_group)) %>% ungroup()
}

# Rename fields to adapt the CAMD crosswalk to the naming conventions used elsewhere in the model, calculate the
# modified prime mover with CCs aggregated, and generate unit-level estimates of capacity from both CAMD and EIA
# nameplate capacity data

CAMD_crosswalk <- CAMD_crosswalk %>%
  rename(
    "plant_id_cems" = "CAMD_PLANT_ID",
    "unit_id_cems" = "CAMD_UNIT_ID",
    "generator_id_cems" = "CAMD_GENERATOR_ID",
    "plant_id_eia" = "EIA_PLANT_ID",
    "unit_code" = "EIA_UNIT_CODE",
    "generator_id" = "EIA_GENERATOR_ID",
    "capacity_cems" = "CAMD_NAMEPLATE_CAPACITY",
    "capacity_eia" = "EIA_NAMEPLATE_CAPACITY_HIST",
    "Prime_hist" = "EIA_UNIT_TYPE",
    "report_year" = "capacity_year"
  ) %>%
  mutate(
    prime_mover =
      case_when(
        Prime_hist == "CA" ~ "CC",
        Prime_hist == "CT" ~ "CC",
        Prime_hist == "CS" ~ "CC",
        TRUE ~ Prime_hist
      ),
    unit_gen_code = ifelse(is.na(unit_code) | !(prime_mover=="CC"),generator_id,unit_code)
  ) %>%
  group_by(CAMD_group,report_year) %>%
  mutate(
    CEMS_nunits = n_distinct(unit_id_cems),
    CEMS_ngens = n_distinct(generator_id_cems,plant_id_cems)
  ) %>% ungroup() %>%
  group_by(plant_id_cems,unit_id_cems,report_year) %>%
  mutate(
    CEMS_n_EIAgens_pu = n_distinct(plant_id_eia,generator_id)
  ) %>% ungroup()

CAMD_crosswalk_CEMS <- CAMD_crosswalk %>%
  select(c(
    plant_id_cems,
    #unit_id_cems,
    generator_id_cems,
    capacity_cems,
    CAMD_group,
    report_year
  )) %>% distinct() %>%
  group_by(CAMD_group,report_year) %>%
  mutate(
    capacity_CAMD_group = sum(capacity_cems, na.rm = TRUE),
    #CAMD_nunits = n_distinct(unit_id_cems),
    CAMD_ngens = n_distinct(generator_id_cems,plant_id_cems)
  ) %>% ungroup() #%>%
  #group_by(CAMD_group,plant_id_cems,unit_id_cems,report_year) %>%
  #mutate(
  #  capacity_cems_unit = sum(capacity_cems, na.rm = TRUE)
  #) %>% ungroup()

#CAMD_crosswalk_CEMS_unit <- CAMD_crosswalk_CEMS %>%
#  select(c(
#    plant_id_cems,
#    unit_id_cems,
#    capacity_cems_unit,
#    capacity_CAMD_group,
#    CAMD_nunits,
#    CAMD_ngens,
#    CAMD_group,
#    report_year
#  )) %>% distinct()

CAMD_crosswalk_CAMD_group <- CAMD_crosswalk_CEMS %>%
  select(c(CAMD_group,capacity_CAMD_group,CAMD_ngens,report_year)) %>% distinct()

CAMD_crosswalk_EIA <- CAMD_crosswalk %>%
  select(c(
    plant_id_eia,
    unit_gen_code,
    generator_id,
    capacity_eia,
    Prime_hist,
    prime_mover,
    CAMD_group,
    report_year
  )) %>% distinct() %>%
  group_by(CAMD_group,report_year) %>%
  mutate(
    capacity_CAMD_group_eia = sum(capacity_eia, na.rm = TRUE),
    CAMD_group_nunit_gen_codes = n_distinct(plant_id_eia,unit_gen_code),
    CAMD_group_neia_gens = n_distinct(generator_id,plant_id_eia)
  ) %>% ungroup() %>%
  group_by(CAMD_group,report_year,Prime_hist) %>%
  mutate(
    capacity_CAMD_group_eia_Prime_hist = sum(capacity_eia, na.rm = TRUE),
    CAMD_group_neia_gens_Prime_hist = n_distinct(generator_id,plant_id_eia)
  ) %>% ungroup() %>%
  left_join(CAMD_crosswalk_CAMD_group, by = c(
    "CAMD_group" = "CAMD_group",
    "report_year" = "report_year"
  )) %>%
  mutate(
    capacity_ratio = capacity_CAMD_group_eia/capacity_CAMD_group
  )

# We now read in CEMS gross generation, MMBTU, and starts data. Then, we filter out CEMS data in which either the CAMD and EIA capacity data
# significantly disagree or for which the CEMS reported max generation suggests that the crosswalk is likely missing generators
# or may just be erroneous (due to exceeding physical capacity bounds). Finally, we group the CEMS data by CAMD_group, and then create a version of
# the CAMD_Crosswalk that only includes unit-months for which filtered CEMS data is available.

CEMS_monthly_data <- read_parquet("camd_unit_starts_ms.parquet") %>%
  rename(report_year = year, report_month = month) %>%
  select(-c(report_date)) %>%
  left_join(CAMD_crosswalk %>% select(c(
    plant_id_cems,
    unit_id_cems,
    report_year,
    capacity_cems,
    capacity_eia,
    capacity_cems_unit,
    capacity_eia_unit,
    CAMD_group
  )) %>% distinct(), by = c(
    "plant_id_cems" = "plant_id_cems",
    "unit_id_cems" = "unit_id_cems",
    "report_year" = "report_year"
  )) %>%
  group_by(plant_id_cems,unit_id_cems,report_year) %>%
  mutate(
    gross_gen_max = ifelse(gross_gen_max < 0.9 * capacity_eia_unit,max(ifelse(gross_gen_max > capacity_eia_unit * 1.2,0,gross_gen_max)),gross_gen_max)
  ) %>% ungroup() %>%
  group_by(plant_id_cems,unit_id_cems) %>%
  mutate(
    gross_gen_max = ifelse(gross_gen_max < 0.9 * capacity_eia_unit,max(ifelse(gross_gen_max > capacity_eia_unit * 1.2,0,gross_gen_max)),gross_gen_max)
  ) %>% ungroup() %>%
  mutate(
    capacity_cems_test = gross_gen_max / capacity_cems_unit,
    capacity_eia_test = gross_gen_max / capacity_eia_unit,
    eia_cems_test = capacity_eia_unit / capacity_cems_unit
  )

CEMS_monthly_data_filtered <- CEMS_monthly_data %>%
  filter((capacity_eia_test<=1.2) & (eia_cems_test>=0.8) & (eia_cems_test<=1.2))

CEMS_monthly_data_grouped <- CEMS_monthly_data_filtered %>%
  group_by(CAMD_group,report_month,report_year) %>%
  summarize(
    CEMS_ngens = n(),
    gen_starts = sum(gen_starts),
    fuel_starts = sum(fuel_starts),
    gross_gen = sum(gross_gen),
    gross_gen_max = sum(gross_gen_max),
    capacity_cems = sum(capacity_cems),
    capacity_eia = sum(capacity_eia),
    heat_in_mmbtu = sum(heat_in_mmbtu),
    co2_tons = sum(co2_tons)
  ) %>% ungroup() %>% distinct()

CAMD_group_crosswalk <- CAMD_crosswalk %>%
  inner_join(CEMS_monthly_data_filtered, by = c(
    "plant_id_cems" = "plant_id_cems",
    "unit_id_cems" = "unit_id_cems",
    "report_year" = "report_year",
    "CAMD_group" = "CAMD_group"
  )) %>%
  select(c(
    CAMD_group,
    plant_id_eia,
    report_year,
    report_month,
    prime_mover,
    generator_id
  )) %>% distinct()

# We now sum the net generation from EIA 923 generator-level monthly data to the CAMD_group level, and merge that data in to
# the grouped CEMS data, only merging in if the number of generators on the EIA and the CEMS side match.

gen_monthly_CAMD_group <- gen_monthly_net_mwh %>%
  left_join(CAMD_group_crosswalk, by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year",
    "report_month" = "report_month"
  )) %>%
  group_by(CAMD_group,report_year,report_month) %>%
  summarize(
    EIA_ngens = sum(!is.na(generator_id)),
    n_primes = n_distinct(prime_mover),
    prime_mover = ifelse(n_primes==1,prime_mover,""),
    net_generation_mwh = sum(net_generation_mwh, na.rm = TRUE)
  ) %>% ungroup() %>% distinct()

CEMS_monthly_data_grouped <- CEMS_monthly_data_grouped %>%
  left_join(gen_monthly_CAMD_group, by = c(
    "CAMD_group" = "CAMD_group",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "CEMS_ngens" = "EIA_ngens"
  )) %>%
  mutate(
    month_hours = unname(24*days_in_month(ymd(report_year*10000+report_month*100+1))),
    parasitic_load = gross_gen - net_generation_mwh,
    parasitic_CF = parasitic_load / (capacity_eia * month_hours),
    gross_CF = gross_gen / (capacity_eia * month_hours)
  )

sum((CEMS_monthly_data_grouped$parasitic_CF >=(0)) &
      (CEMS_monthly_data_grouped$parasitic_CF <=(0.1)) &
      (CEMS_monthly_data_grouped$prime_mover=="GT"), na.rm = TRUE)/
  sum((CEMS_monthly_data_grouped$prime_mover=="GT"), na.rm = TRUE)


# Read in historical 860 data to get unit nameplate capacity and get the fuels primarily used by each generator. Here, we normalize data
# for all CCs, by grouping all generators of a CC unit sharing the same unit_code. Note that we only do this for generators reporting
# prime movers as CA, CT, and CS - we do not group together ST, GT, or IC generators with reported unit codes.

historic_860_essentials <- read_parquet("../patio_data/historic_eia_860_gen_essentials.parquet")

historic_860_essentials <- historic_860_essentials %>%
  rename(
    "plant_id_eia" = "EIA_PLANT_ID",
    "generator_id" = "EIA_GENERATOR_ID",
    "unit_code" = "EIA_UNIT_CODE",
    "Prime_hist" = "EIA_UNIT_TYPE_HIST",
    "capacity_hist" = "EIA_NAMEPLATE_CAPACITY_HIST",
    "energy_source_code_hist_1" = "EIA_FUEL_TYPE_1_HIST",
    "energy_source_code_hist_2" = "EIA_FUEL_TYPE_2_HIST",
    "energy_source_code_hist_3" = "EIA_FUEL_TYPE_3_HIST",
    "energy_source_code_hist_4" = "EIA_FUEL_TYPE_4_HIST",
    "energy_source_code_hist_5" = "EIA_FUEL_TYPE_5_HIST",
    "energy_source_code_hist_6" = "EIA_FUEL_TYPE_6_HIST",
    "report_year" = "capacity_year"
  ) %>%
  mutate(
    prime_mover_code_hist =
      case_when(
        Prime_hist == "CA" ~ "CC",
        Prime_hist == "CT" ~ "CC",
        Prime_hist == "CS" ~ "CC",
        TRUE ~ Prime_hist
      ),
    unit_gen_code = ifelse(is.na(unit_code) | !(prime_mover_code_hist=="CC"),generator_id,unit_code)
  )

# Next, we pivot the fuel data longer and generate a unit-level long data set of all plants, unit_gen_codes and fuels by year
# which allows us to treat generators in CCs as sharing all the available fuels for further analysis, and create a unit_gen_code map
# We then use this map iteratively to generate connected groups of unit_gens that share the same plant_id, prime mover code, and
# an energy source code, labeled by "ppf_group".

historic_860_ppf <- historic_860_essentials %>%
  pivot_longer(cols = starts_with("energy_source_code_hist_"),
               names_to = "energy_source_code_num",
               names_prefix = "energy_source_code_hist_",
               names_transform = list(energy_source_code_num = as.integer),
               values_to = "energy_source_code",
               values_drop_na = TRUE) %>%
  filter(#(energy_source_code_num==7) |
    !(is.na(energy_source_code) | ((energy_source_code=="WH") & (Prime_hist=="CA")))) %>%
  mutate(energy_source_code_num = ifelse(energy_source_code_num == 1 | energy_source_code_num == 2,
                                         energy_source_code_num, NA)) %>%
  select(-c(generator_id,unit_code,Prime_hist,capacity_hist)) %>% distinct() %>%
  add_count(plant_id_eia,unit_gen_code,energy_source_code,report_year, name = "duplicated") %>%
  filter(!((duplicated>1) & is.na(energy_source_code_num))) %>% select(-c(duplicated)) %>%
  add_count(plant_id_eia,unit_gen_code,energy_source_code,report_year, name = "duplicated") %>%
  filter(!((duplicated>1) & (energy_source_code_num==2))) %>% select(-c(duplicated)) %>%
  mutate(historic_ppf_index = row_number()) %>%
  group_by(report_year, plant_id_eia, unit_gen_code) %>%
  mutate(
    ppf_group = max(historic_ppf_index)
  ) %>% select(-c(historic_ppf_index)) %>% ungroup() %>% distinct()

for (iter in 1:20) {
  historic_860_ppf <- historic_860_ppf %>%
    group_by(report_year, plant_id_eia, unit_gen_code) %>%
    mutate(ppf_group = max(ppf_group)) %>% ungroup() %>%
    group_by(report_year, plant_id_eia, prime_mover_code_hist, energy_source_code) %>%
    mutate(ppf_group = max(ppf_group)) %>% ungroup()
}

unit_gen_code_map <- historic_860_essentials %>%
  select(c(plant_id_eia,generator_id,report_year,unit_gen_code)) %>% distinct()

# Now, we read in 923 generation and fuel data and calculate fuel and generation totals at the ppf_group level.
# This will allow us to later match this data to all relevant generators to estimate CF, heat rate, and fuel
# fractions for all units without boiler / generator level data

gen_fuel_923 <- read_parquet("gen_fuel_923_historical.parquet")
gen_fuel_923 <- gen_fuel_923 %>%
  select(c(
    ppf_index = `__index_level_0__`,
    report_year,
    report_month,
    plant_id_eia,
    prime_mover_code,
    energy_source_code,
    mmbtu_923gf = fuel_consumed_for_electricity_mmbtu,
    net_generation_mwh_923gf = net_generation_mwh
  )) %>%
  mutate(#report_date = report_date + hours(8),
    #report_year = year(report_date),
    #report_month = month(report_date),
    #report_date = round_date(report_date,
    #                        unit = "day")
    #Prime = prime_mover_code,
    prime_mover_code =
      case_when(
        prime_mover_code == "CA" ~ "CC",
        prime_mover_code == "CT" ~ "CC",
        prime_mover_code == "CS" ~ "CC",
        TRUE ~ prime_mover_code
      )
  ) %>%
  #select(-c(report_date)) %>%
  left_join(fuel_map, by = c("energy_source_code" = "energy_source_code")) %>%
  left_join(historic_860_ppf %>%
              select(c(plant_id_eia,report_year,prime_mover_code_hist,energy_source_code,ppf_group)) %>%
              distinct(),by=c(
                "plant_id_eia" = "plant_id_eia",
                "prime_mover_code" = "prime_mover_code_hist",
                "report_year" = "report_year",
                "energy_source_code" = "energy_source_code"
              )) %>%
  group_by(report_month, report_year, plant_id_eia, ppf_group) %>%
  mutate(
    mmbtu_pg = sum(mmbtu_923gf),
    net_generation_mwh_pg = sum(net_generation_mwh_923gf)
  ) %>% ungroup() %>%
  group_by(report_month, report_year, plant_id_eia, ppf_group, energy_source_code) %>%
  mutate(
    mmbtu_pg_f = sum(mmbtu_923gf),
    net_generation_mwh_pg_f = sum(net_generation_mwh_923gf)
  ) %>% ungroup() %>%
  group_by(report_month, report_year, plant_id_eia, ppf_group, fuel_group_code) %>%
  mutate(
    mmbtu_pg_fg = sum(mmbtu_923gf),
    net_generation_mwh_pg_fg = sum(net_generation_mwh_923gf)
  ) %>% ungroup() %>%
  select(-c(mmbtu_923gf,net_generation_mwh_923gf,ppf_index)) %>% distinct() #%>%
#  mutate(ppf_index = row_number())

historic_ppf <- historic_860_ppf %>%
  left_join(gen_fuel_923, by = c(
    "plant_id_eia" = "plant_id_eia",
    "prime_mover_code_hist" = "prime_mover_code",
    "report_year" = "report_year",
    "energy_source_code" = "energy_source_code",
    "ppf_group" = "ppf_group"
  ))

# Read in historical boiler-generator associations to assist in allocating historical boiler-level fuel consumption to generators,
# link in unit_gen_code, and impute incomplete bga data from 2008 by mapping in 2009 bga to 2008

bga <- read_parquet("boiler_gen_assoc_860_historical.parquet")

bga <- bga %>%
  #mutate(
  #  report_year = year(report_date)
  #) %>%
  select(c(
    report_year,
    plant_id_eia,
    generator_id,
    unit_id_eia,
    boiler_id
  ))

bga <- bga %>% rbind(bga %>% filter(report_year==2009) %>% mutate(report_year=2008)) %>% distinct()

bga <- bga %>%
  mutate(
    no_match = 1,
    bga_index = row_number(),
    unit_id_eia = ifelse(plant_id_eia == 50973 & (unit_id_eia == "BLK2" | unit_id_eia == "BLK3"),"BLK1",unit_id_eia),
    #boiler_unit_gen_code = ifelse(is.na(unit_id_eia),generator_id,unit_id_eia),
  ) %>% filter(!(plant_id_eia == 10725 & boiler_id == "HRSG1" & unit_id_eia == "F801")) %>%
  left_join(unit_gen_code_map,by=c(
    "report_year" = "report_year",
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id"
  )) %>% select(-c(unit_id_eia,generator_id)) %>% distinct()

# Read in historical monthly boiler-level fuel consumption, and map it to all relevant generators using the bga associations. We attempt to fix
# the missing historical associations using 2021 data or just assuming that the unit_gen_code and the boiler_id are the same. Then,
# we generate maximal, connected boiler groups associated with a maximal set of generators iteratively.

bga_rev <- read_parquet("boiler_fuel_923_historical.parquet") %>%
  #mutate(
  #  report_year = year(report_date),
  #  report_month = month(report_date),
  #) %>%
  select(c(
    bf_index = `__index_level_0__`,
    report_year,
    report_month,
    plant_id_eia,
    boiler_id,
    boiler_energy_source_code = energy_source_code,
    mmbtu_bm_pbf = fuel_consumed_mmbtu
  )) %>%
  left_join(bga, by = c(
    "plant_id_eia" = "plant_id_eia",
    "boiler_id" = "boiler_id",
    "report_year" = "report_year"
  ))  %>% mutate(no_match = is.na(unit_gen_code)) %>%
  left_join(bga %>% filter(report_year == 2021) %>% select(-c(report_year,bga_index)) %>%
              rename("unit_gen_code_2021" = "unit_gen_code"), by = c(
                "plant_id_eia" = "plant_id_eia",
                "no_match" = "no_match",
                "boiler_id" = "boiler_id"
              )) %>%
  mutate(
    bga_index = ifelse(is.na(unit_gen_code),max(bga_index, na.rm = TRUE)+row_number(),bga_index),
    unit_gen_code = ifelse(is.na(unit_gen_code),ifelse(is.na(unit_gen_code_2021),boiler_id,unit_gen_code_2021),unit_gen_code)
  ) %>%
  select(c(
    report_year,
    plant_id_eia,
    boiler_id,
    unit_gen_code,
    bga_index
  )) %>% distinct() %>%
  group_by(report_year, plant_id_eia, unit_gen_code) %>%
  mutate(
    boiler_group = max(bga_index)
  ) %>% select(-c(bga_index)) %>% ungroup() %>% distinct()

for (iter in 1:20) {
  bga_rev <- bga_rev %>%
    group_by(report_year, plant_id_eia, unit_gen_code) %>%
    mutate(boiler_group = max(boiler_group)) %>% ungroup() %>%
    group_by(report_year, plant_id_eia, boiler_id) %>%
    mutate(boiler_group = max(boiler_group)) %>% ungroup()
}

# Process generator-level net generation data to the unit_gen_code level for integration with boiler data

unit_gen_net_mwh <- gen_monthly_net_mwh%>%
  left_join(unit_gen_code_map %>%
              add_count(report_year,plant_id_eia,unit_gen_code, name = "gen_count"),by=c(
                "report_year" = "report_year",
                "plant_id_eia" = "plant_id_eia",
                "generator_id" = "generator_id"
              )) %>%
  add_count(report_year, report_month, plant_id_eia, unit_gen_code, name = "reporting_gen_count") %>%
  group_by(report_year, report_month, plant_id_eia, unit_gen_code) %>%
  summarize(
    #reporting_gen_count = reporting_gen_count,
    #gen_count = gen_count,
    net_generation_mwh = #ifelse(reporting_gen_count == gen_count,
      sum(net_generation_mwh)#,NA)
  ) %>% ungroup() %>% distinct()

# Now, we pull use the revised boiler-unit_gen associations and pull back in the monthly boiler data along with historical monthly
# unit_gen_code level net generation data, and aggregate both fuel and net_gen data by unit_gen_code and boiler_group. We do this
# because we are only able to reliably utilize data for heat rate and fuel fraction calculations if all the boilers and generators
# associated to each other are aggregated.

boiler_monthly_mmbtu <- read_parquet("boiler_fuel_923_historical.parquet") %>%
  #mutate(
  #  report_year = year(report_date),
  #  report_month = month(report_date),
  #) %>%
  select(c(
    bf_index = `__index_level_0__`,
    report_year,
    report_month,
    plant_id_eia,
    boiler_id,
    boiler_energy_source_code = energy_source_code,
    mmbtu_bm_pbf = fuel_consumed_mmbtu
  )) %>%
  left_join(bga_rev, by = c(
    "plant_id_eia" = "plant_id_eia",
    "boiler_id" = "boiler_id",
    "report_year" = "report_year"
  ))  %>%
  left_join(unit_gen_net_mwh, by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year",
    "report_month" = "report_month"
  )) %>%
  left_join(fuel_map, by = c("boiler_energy_source_code" = "energy_source_code")) %>%
  rename("boiler_fuel_group_code" = "fuel_group_code") %>%
  group_by(report_month, report_year, plant_id_eia, unit_gen_code) %>%
  mutate(
    mmbtu_pug = sum(mmbtu_bm_pbf),
    bfs_count = n()
  ) %>% ungroup() %>%
  add_count(report_month, report_year, plant_id_eia, boiler_id, name = "bug_count") %>%
  add_count(bf_index, name = "bfug_count") %>%
  group_by(report_month, report_year, boiler_group) %>%
  mutate(
    mmbtu_bg = ifelse(bfug_count>0,sum(mmbtu_bm_pbf/bfug_count),NA),
    net_generation_mwh_bg = ifelse(bfs_count>0,sum(net_generation_mwh/bfs_count),NA)
  ) %>% ungroup() %>%
  group_by(report_month, report_year, plant_id_eia, boiler_group, boiler_fuel_group_code) %>%
  mutate(
    mmbtu_bg_fg = ifelse(bfug_count>0,sum(mmbtu_bm_pbf/bfug_count),NA)
  ) %>%  ungroup() %>% select(-c(boiler_fuel_group_code)) %>%
  group_by(report_month, report_year, plant_id_eia, boiler_group, boiler_energy_source_code) %>%
  mutate(
    mmbtu_bg_f = ifelse(bfug_count>0,sum(mmbtu_bm_pbf/bfug_count),NA)
  ) %>% rename("net_generation_mwh_bm_pug" = "net_generation_mwh") %>% ungroup() %>%
  select(-c(
    bf_index,
    boiler_id,
    mmbtu_bm_pbf,
    bfs_count,
    bug_count,
    bfug_count
  )) %>% distinct()

# Pull in CEMS data on monthly gross generation and estimating the split between the top two fuels used in each month
# for each reporting EIA generator

fuel_splits_and_starts <- read_parquet("fuel_split_and_starts.parquet")

fuel_splits_and_starts <- fuel_splits_and_starts %>%
  mutate(
    datetime = datetime + hours(8),
    datetime = round_date(datetime,unit = "day"),
    report_year = year(datetime),
    report_month = month(datetime),
  ) %>% select(-c(datetime)) %>%
  pivot_longer(cols = starts_with("energy_source_code_"),
               names_to = "energy_source_code_fss_num",
               names_prefix = "energy_source_code_",
               names_transform = list(energy_source_code_fss_num = as.integer),
               values_to = "energy_source_code",
               values_drop_na = TRUE) %>%
  mutate(mmbtu_fss_f = ifelse(energy_source_code_fss_num==1,fuel_1_mmbtu,fuel_2_mmbtu)) %>%
  select(-c(fuel_1_mmbtu,fuel_2_mmbtu)) %>%
  filter(!(energy_source_code==""))

fss_map <- fuel_splits_and_starts %>%
  select(c(report_year,plant_id_eia,energy_source_code,generator_id)) %>%
  left_join(unit_gen_code_map,by = c(
    "report_year" = "report_year",
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id"
  )) %>% select(-c(generator_id)) %>% distinct()

# Finally, we combine all the fuel sources linked to any unit_gens and create a master map of all associated fuels to
# any unit_gen in any given year.

unit_gen_fuel_map <- boiler_monthly_mmbtu %>%
  select(c(
    report_year,
    plant_id_eia,
    boiler_energy_source_code,
    unit_gen_code
  )) %>% distinct() %>%
  rename("energy_source_code" = "boiler_energy_source_code") %>%
  rbind(historic_860_ppf %>% select(c(
    report_year,
    plant_id_eia,
    energy_source_code,
    unit_gen_code
  ))) %>% distinct() %>%
  rbind(fss_map) %>% distinct()


# Now, we move to a multi-step process to link historical 860, 923, and CEMS data to estimate capacity factors,
# fuel fractions, and heat rates at various levels of aggregation. Since all the generation and fuel data has been
# tagged and grouped into connected components, with aggregate generation and mmbtu calculations already complete,
# This process is relatively straightforward.

# Step 1: Start by pulling in the 923 generator level data and calculating capacity fractions, and then pulling in
# the fuel source independent data from 923 gen fuel, 923 boiler fuel, and CEMS.

historic_860_923_data <- units_present_hist %>%
  inner_join(historic_860_essentials %>% select(-c(
    energy_source_code_hist_1,
    energy_source_code_hist_2,
    energy_source_code_hist_3,
    energy_source_code_hist_4,
    energy_source_code_hist_5,
    energy_source_code_hist_6#,
    #energy_source_code_hist_7
  )), by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year"
  )) %>%
  left_join(gen_monthly_net_mwh, by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year",
    "report_month" = "report_month"
  )) %>%
  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
  mutate(
    capacity_hist_pug = sum(capacity_hist),
    capacity_frac_pug = capacity_hist / capacity_hist_pug
  ) %>% ungroup() %>%
  left_join(boiler_monthly_mmbtu %>% select(-c(
    mmbtu_bg_fg,
    mmbtu_bg_f,
    boiler_energy_source_code,
  )) %>% distinct(), by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year",
    "report_month" = "report_month"
  )) %>%
  left_join(historic_ppf %>% select(-c(
    prime_mover_code_hist,
    fuel_group_code,
    energy_source_code_num,
    energy_source_code,
    mmbtu_pg_f,
    net_generation_mwh_pg_f,
    mmbtu_pg_fg,
    net_generation_mwh_pg_fg
  )) %>% distinct(), by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year",
    "report_month" = "report_month"
  )) %>%
  left_join(fuel_splits_and_starts %>% select(-c(
    prime_mover_code,
    energy_source_code_fss_num,
    energy_source_code,
    mmbtu_fss_f
  )) %>% distinct(), by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year",
    "report_month" = "report_month"
  ))

# Step 2: Next, we pull in the fuel-dependent 923 boiler fuel and generation fuel data as well as the CEMS starts and stops.

historic_860_923_data <- historic_860_923_data %>%
  left_join(unit_gen_fuel_map, by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year"
  )) %>%
  left_join(boiler_monthly_mmbtu %>% select(-c(
    boiler_group,
    net_generation_mwh_bm_pug,
    mmbtu_pug,
    mmbtu_bg,
    net_generation_mwh_bg
  )), by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "energy_source_code" = "boiler_energy_source_code"
  )) %>%
  left_join(historic_ppf %>% select(-c(
    prime_mover_code_hist,
    fuel_group_code,
    ppf_group,
    mmbtu_pg,
    net_generation_mwh_pg
  )), by = c(
    "plant_id_eia" = "plant_id_eia",
    "unit_gen_code" = "unit_gen_code",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "energy_source_code" = "energy_source_code"
  )) %>%
  left_join(fuel_splits_and_starts %>% select(-c(
    prime_mover_code,
    gross_gen,
    heat_in_mmbtu,
    co2_tons,
    gen_starts,
    fuel_starts
  )), by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "energy_source_code" = "energy_source_code"
  )) %>%
  add_count(plant_id_eia,generator_id,report_year,report_month, name = "n_energy_sources")

# Step 3: We calculate capacity fractions for the set of generators both included in a boiler_group or ppf_group.
# Note that the 923 generator fuel data has been matched to the generator reported energy source data, but not to
# reported starter fuels to focus on the dominant units / generators that use any given fuel.

# This is not true for the boiler fuel data, which likely includes all fuels used by a boiler. However, since those fuels are associated by boiler,
# the resulting total fuel fractions and consumption are likely to be a bit more accurate in the boiler groups. Further, as we have noted earlier,
# all boiler and ppf associations have been aggregated to have the same associations for all generators for multi-generator CCs in order to allow
# us to analyze CCs moving forward based on allocation of unit-level CFs, heat rates. The CC aggregation is necessary for our
# economic analysis, which will treat all the generators in an given CC as a single unit.

# Finally, we identify the most granular data available for each generator and use it to estimate net generation (allocated by capacity fraction),
# and mmbtu by fuel (allocated by net generation among generators)

historic_860_923_data <- historic_860_923_data %>%
  group_by(boiler_group,report_year,report_month) %>%
  mutate(
    #n_plants_bg = ifelse(is.na(boiler_group),NA,n_distinct(plant_id_eia)),
    #n_primes_bg = ifelse(is.na(boiler_group),NA,n_distinct(prime_mover_code_hist)),
    n_unit_gens_bg = ifelse(is.na(boiler_group),NA,n_distinct(unit_gen_code)),
    capacity_hist_bg = ifelse(is.na(boiler_group),NA,sum(capacity_hist/n_energy_sources)),
    capacity_frac_bg = ifelse(is.na(boiler_group),NA,capacity_hist/capacity_hist_bg),
  ) %>% ungroup() %>%
  group_by(ppf_group,report_year,report_month) %>%
  mutate(
    n_unit_gens_pg = ifelse(is.na(ppf_group),NA,n_distinct(unit_gen_code)),
    capacity_hist_pg = ifelse(is.na(ppf_group),NA,sum(capacity_hist/n_energy_sources)),
    capacity_frac_pg = ifelse(is.na(ppf_group),NA,capacity_hist/capacity_hist_pg),
  ) %>% ungroup() %>%
  group_by(boiler_group) %>%
  mutate(
    gross_gen_raw_bg = ifelse(is.na(boiler_group),NA,sum(gross_gen/n_energy_sources, na.rm = TRUE)),
    parasitic_load_per_MW_bg = (gross_gen_raw_bg -
                                  sum(ifelse(is.na(gross_gen),0,net_generation_mwh_bg * capacity_frac_bg / n_energy_sources)))/
      sum(ifelse(is.na(gross_gen),0,capacity_hist / n_energy_sources))
  ) %>% ungroup() %>%
  group_by(ppf_group) %>%
  mutate(
    gross_gen_raw_pg = ifelse(is.na(ppf_group),NA,sum(gross_gen/n_energy_sources, na.rm = TRUE)),
    parasitic_load_per_MW_pg = (gross_gen_raw_pg -
                                  sum(ifelse(is.na(gross_gen),0,net_generation_mwh_pg * capacity_frac_pg / n_energy_sources)))/
      sum(ifelse(is.na(gross_gen),0,capacity_hist / n_energy_sources))
  ) %>% ungroup() %>%
  left_join(fuel_map, by = c("energy_source_code" = "energy_source_code")) %>%
  mutate(
    net_generation_mwh_final = case_when(
      !is.na(net_generation_mwh) ~ net_generation_mwh,
      !is.na(net_generation_mwh_bm_pug) ~ net_generation_mwh_bm_pug * capacity_frac_pug,
      !is.na(net_generation_mwh_bg) & n_unit_gens_bg <= n_unit_gens_pg ~ net_generation_mwh_bg * capacity_frac_bg,
      !is.na(net_generation_mwh_pg) ~ net_generation_mwh_pg * capacity_frac_pg
    )
  ) %>%
  mutate(
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW_bg),parasitic_load_per_MW_pg,parasitic_load_per_MW_bg),
    parasitic_load_per_MW = ifelse(parasitic_load_per_MW>0,parasitic_load_per_MW,NA),
    parasitic_load = parasitic_load_per_MW * capacity_hist
  ) %>%
  add_count(fuel_group_code,plant_id_eia,generator_id,report_year,report_month, name = "n_fuel_group_lines") %>%
  add_count(plant_id_eia,generator_id,report_year,report_month, name = "n_gen_lines") %>%
  group_by(plant_id_eia,prime_mover_code_hist,fuel_group_code) %>%
  mutate(
    fss_capacity_hist = sum(ifelse(is.na(parasitic_load) | n_fuel_group_lines==0,0,
                                   capacity_hist/n_fuel_group_lines)),
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW) | (parasitic_load_per_MW < 0),
                                   sum(ifelse(is.na(parasitic_load) | n_fuel_group_lines==0,0,
                                              parasitic_load/n_fuel_group_lines))/fss_capacity_hist,parasitic_load_per_MW)
  ) %>% ungroup() %>%
  group_by(fuel_group_code,prime_mover_code_hist) %>%
  mutate(
    fss_capacity_hist_all = sum(ifelse(is.na(parasitic_load) | n_fuel_group_lines==0,0,
                                       capacity_hist/n_fuel_group_lines)),
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW) | (parasitic_load_per_MW < 0),
                                   sum(ifelse(is.na(parasitic_load) | n_fuel_group_lines==0,0,
                                              parasitic_load/n_fuel_group_lines))/fss_capacity_hist_all,parasitic_load_per_MW)
  ) %>% ungroup() %>%
  group_by(plant_id_eia,prime_mover_code_hist) %>%
  mutate(
    fss_capacity_hist_all = sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                       capacity_hist/n_gen_lines)),
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW) | (parasitic_load_per_MW < 0),
                                   sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                              parasitic_load/n_gen_lines))/fss_capacity_hist_all,parasitic_load_per_MW)
  ) %>% ungroup() %>%
  group_by(plant_id_eia) %>%
  mutate(
    fss_capacity_hist_all = sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                       capacity_hist/n_gen_lines)),
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW) | (parasitic_load_per_MW < 0),
                                   sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                              parasitic_load/n_gen_lines))/fss_capacity_hist_all,parasitic_load_per_MW)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist) %>%
  mutate(
    fss_capacity_hist_all = sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                       capacity_hist/n_gen_lines)),
    parasitic_load_per_MW = ifelse(is.na(parasitic_load_per_MW) | (parasitic_load_per_MW < 0),
                                   sum(ifelse(is.na(parasitic_load) | n_gen_lines==0,0,
                                              parasitic_load/n_gen_lines))/fss_capacity_hist_all,parasitic_load_per_MW)
  ) %>% ungroup() %>%
  mutate(
    parasitic_load = parasitic_load_per_MW * capacity_hist,
    net_generation_mwh_final = ifelse(!is.na(gross_gen) & is.na(net_generation_mwh_final)
                                      & !is.na(parasitic_load) & (gross_gen > parasitic_load),
                                      gross_gen - parasitic_load, net_generation_mwh_final),
    gross_gen_final = net_generation_mwh_final+parasitic_load,
    CF_net = net_generation_mwh_final / (capacity_hist * month_hours),
    CF_gross = gross_gen_final / (capacity_hist * month_hours)
  ) %>%
  group_by(plant_id_eia,boiler_group,report_year,report_month) %>%
  mutate(
    gross_gen_bg = sum(gross_gen_final/n_energy_sources, na.rm = TRUE)
  ) %>% ungroup() %>%
  group_by(plant_id_eia,ppf_group,report_year,report_month) %>%
  mutate(
    gross_gen_pg = sum(gross_gen_final/n_energy_sources, na.rm = TRUE)
  ) %>% ungroup() %>%
  mutate(
    mmbtu_final = case_when(
      !(is.na(mmbtu_bg)) & (gross_gen_bg>0) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg * gross_gen_final/gross_gen_bg,
      !(is.na(mmbtu_pg)) & (gross_gen_pg>0) ~ mmbtu_pg * gross_gen_final/gross_gen_pg,
      !(is.na(mmbtu_bg)) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg * capacity_frac_bg,
      !(is.na(mmbtu_pg)) ~ mmbtu_pg * capacity_frac_pg
    ),
    mmbtu_f_final = case_when(
      !(is.na(mmbtu_bg_f)) & (gross_gen_bg>0) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg_f * gross_gen_final/gross_gen_bg,
      !(is.na(mmbtu_pg_f)) & (gross_gen_pg>0) ~ mmbtu_pg_f * gross_gen_final/gross_gen_pg,
      !(is.na(mmbtu_bg_f)) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg_f * capacity_frac_bg,
      !(is.na(mmbtu_pg_f)) ~ mmbtu_pg_f * capacity_frac_pg
    ),
    mmbtu_fg_final = case_when(
      !(is.na(mmbtu_bg_fg)) & (gross_gen_bg>0) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg_fg * gross_gen_final/gross_gen_bg,
      !(is.na(mmbtu_pg_fg)) & (gross_gen_pg>0) ~ mmbtu_pg_fg * gross_gen_final/gross_gen_pg,
      !(is.na(mmbtu_bg_fg)) & n_unit_gens_bg <= n_unit_gens_pg ~ mmbtu_bg_fg * capacity_frac_bg,
      !(is.na(mmbtu_pg_fg)) ~ mmbtu_pg_fg * capacity_frac_pg
    )
  ) %>%
  left_join(fuel_group_and_emissions_map %>%
              select(c(energy_source_code,co2e_mt_per_mmbtu,co2_mt_per_mmbtu)),
            by=c("energy_source_code" = "energy_source_code")) %>%
  mutate(
    co2e = co2e_mt_per_mmbtu * mmbtu_f_final,
    co2 = co2_mt_per_mmbtu * mmbtu_f_final,
    fuel_frac = ifelse(!is.na(mmbtu_final) & mmbtu_final>0,mmbtu_fg_final / mmbtu_final,0),
    gross_gen_final = ifelse(is.na(mmbtu_final) & (net_generation_mwh_final==0),0,gross_gen_final)
  ) %>%
  group_by(fuel_group_code,plant_id_eia,generator_id,report_year,report_month) %>%
  mutate(
    co2e_fg = sum(co2e, na.rm = TRUE),
    co2_fg = sum(co2, na.rm = TRUE)
  ) %>% ungroup() %>%
  left_join(plant_grouping, by = c("plant_id_eia" = "plant_id_eia")) %>%
  group_by(prime_mover_code_hist,fuel_group_code,report_month,report_year) %>%
  mutate(
    n_bins = floor((sum(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3, na.rm = TRUE))^(1/3)),
    n_bins = ifelse(n_bins==0,n_bins+1,n_bins),
    cap_tile = ntile(ifelse(!((State=="HI") | (State=="AK")) & fuel_frac>=0.3,capacity_hist,NA),n_bins),
    lat_tile = ntile(ifelse(!((State=="HI") | (State=="AK")) & fuel_frac>=0.3,Latitude,NA),n_bins),
    long_tile = ntile(ifelse(!((State=="HI") | (State=="AK")) & fuel_frac>=0.3,Longitude,NA),n_bins)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist,fuel_group_code,report_month,report_year,cap_tile,lat_tile,long_tile) %>%
  mutate(
    cap_sum = sum(ifelse(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                         capacity_hist,NA),na.rm = TRUE),
    CF_net_ave = ifelse(cap_sum>0,sum(
      ifelse(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             net_generation_mwh_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),
    CF_gross_ave = ifelse(cap_sum>0,sum(
      ifelse(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             gross_gen_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),
    gross_sum = sum(ifelse(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                           gross_gen_final,NA),na.rm = TRUE),
    mmbtu_per_mwh_ave = ifelse(gross_sum>0,sum(
      ifelse(!((State=="HI") | (State=="AK") | is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             mmbtu_final,NA),na.rm = TRUE)/gross_sum,NA),
    gen_starts_ave = median(ifelse(!((State=="HI") | (State=="AK") | is.na(gen_starts)) & fuel_frac>=0.3,
                                   gen_starts,NA),na.rm = TRUE),
    fuel_starts_ave = median(ifelse(!((State=="HI") | (State=="AK") | is.na(fuel_starts)) & fuel_frac>=0.3,
                                    fuel_starts,NA),na.rm = TRUE)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist,fuel_group_code,report_month,report_year,State) %>%
  mutate(
    cap_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,capacity_hist,NA),na.rm = TRUE),
    CF_net_ave = ifelse(is.na(CF_net_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                      net_generation_mwh_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_net_ave),
    CF_gross_ave = ifelse(is.na(CF_gross_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                          gross_gen_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_gross_ave),
    gross_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                           gross_gen_final,NA),na.rm = TRUE),
    mmbtu_per_mwh_ave = ifelse(is.na(mmbtu_per_mwh_ave),ifelse(gross_sum>0,sum(
      ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             mmbtu_final,NA),na.rm = TRUE)/gross_sum,NA),mmbtu_per_mwh_ave),
    gen_starts_ave = ifelse(is.na(gen_starts_ave),median(ifelse(!(is.na(gen_starts)) & fuel_frac>=0.3,
                                                                gen_starts,NA),na.rm = TRUE),gen_starts_ave),
    fuel_starts_ave = ifelse(is.na(fuel_starts_ave),median(ifelse(!(is.na(fuel_starts)) & fuel_frac>=0.3,
                                                                  fuel_starts,NA),na.rm = TRUE),fuel_starts_ave)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist,fuel_group_code,report_month,report_year,Balancing_Authority_Code) %>%
  mutate(
    cap_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,capacity_hist,NA),na.rm = TRUE),
    CF_net_ave = ifelse(is.na(CF_net_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                      net_generation_mwh_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_net_ave),
    CF_gross_ave = ifelse(is.na(CF_gross_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                          gross_gen_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_gross_ave),
    gross_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                           gross_gen_final,NA),na.rm = TRUE),
    mmbtu_per_mwh_ave = ifelse(is.na(mmbtu_per_mwh_ave),ifelse(gross_sum>0,sum(
      ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             mmbtu_final,NA),na.rm = TRUE)/gross_sum,NA),mmbtu_per_mwh_ave),
    gen_starts_ave = ifelse(is.na(gen_starts_ave),median(ifelse(!(is.na(gen_starts)) & fuel_frac>=0.3,
                                                                gen_starts,NA),na.rm = TRUE),gen_starts_ave),
    fuel_starts_ave = ifelse(is.na(fuel_starts_ave),median(ifelse(!(is.na(fuel_starts)) & fuel_frac>=0.3,
                                                                  fuel_starts,NA),na.rm = TRUE),fuel_starts_ave)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist,fuel_group_code,report_month,report_year) %>%
  mutate(
    cap_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,capacity_hist,NA),na.rm = TRUE),
    CF_net_ave = ifelse(is.na(CF_net_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                      net_generation_mwh_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_net_ave),
    CF_gross_ave = ifelse(is.na(CF_gross_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                                                                          gross_gen_final,NA),na.rm = TRUE)/(cap_sum*month_hours),NA),CF_gross_ave),
    gross_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                           gross_gen_final,NA),na.rm = TRUE),
    mmbtu_per_mwh_ave = ifelse(is.na(mmbtu_per_mwh_ave),ifelse(gross_sum>0,sum(
      ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             mmbtu_final,NA),na.rm = TRUE)/gross_sum,NA),mmbtu_per_mwh_ave),
    gen_starts_ave = ifelse(is.na(gen_starts_ave),median(ifelse(!(is.na(gen_starts)) & fuel_frac>=0.3,
                                                                gen_starts,NA),na.rm = TRUE),gen_starts_ave),
    fuel_starts_ave = ifelse(is.na(fuel_starts_ave),median(ifelse(!(is.na(fuel_starts)) & fuel_frac>=0.3,
                                                                  fuel_starts,NA),na.rm = TRUE),fuel_starts_ave)
  ) %>% ungroup() %>%
  group_by(prime_mover_code_hist,report_month,report_year) %>%
  mutate(
    cap_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)),capacity_hist,NA)/n_energy_sources,na.rm = TRUE),
    CF_net_ave = ifelse(is.na(CF_net_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)),
                                                                      net_generation_mwh_final,NA),na.rm = TRUE)/(cap_sum*month_hours*n_energy_sources),NA),CF_net_ave),
    CF_gross_ave = ifelse(is.na(CF_gross_ave),ifelse(cap_sum>0,sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)),
                                                                          gross_gen_final,NA),na.rm = TRUE)/(cap_sum*month_hours*n_energy_sources),NA),CF_gross_ave),
    gross_sum = sum(ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
                           gross_gen_final,NA),na.rm = TRUE),
    mmbtu_per_mwh_ave = ifelse(is.na(mmbtu_per_mwh_ave),ifelse(gross_sum>0,sum(
      ifelse(!(is.na(net_generation_mwh_final) | is.na(gross_gen_final)) & fuel_frac>=0.3,
             mmbtu_final,NA),na.rm = TRUE)/gross_sum,NA),mmbtu_per_mwh_ave),
    gen_starts_ave = ifelse(is.na(gen_starts_ave),median(ifelse(!(is.na(gen_starts)),
                                                                gen_starts,NA),na.rm = TRUE),gen_starts_ave),
    fuel_starts_ave = ifelse(is.na(fuel_starts_ave),median(ifelse(!(is.na(fuel_starts)),
                                                                  fuel_starts,NA),na.rm = TRUE),fuel_starts_ave)
  ) %>% ungroup()
# %>% filter(!is.na(net_generation_mwh_final))

# Step 4: Now, we merge in fuel costs and save the complete data set for other potential applications.

historic_860_923_data <- historic_860_923_data %>%
  left_join(final_fuel_costs, by = c(
    "plant_id_eia" = "plant_id_eia",
    "prime_mover_code_hist" = "prime_mover_code",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "fuel_group_code" = "fuel_group_code"
  ))

write_parquet(historic_860_923_data,"complete_historic_unit_data.parquet")
historic_860_923_data <- read_parquet("complete_historic_unit_data.parquet")

# Generate parasitic load estimates by plant-prime-fuel.

parasitic_load_est <- historic_860_923_data %>%
  select(c(
    plant_id_eia,
    generator_id,
    prime_mover_code_hist,
    report_year,
    report_month,
    parasitic_load,
    capacity_hist,
    fuel_group_code,
    net_generation_mwh_final,
    gross_gen_final,
    fuel_frac,
    month_hours
  )) %>%
  group_by(plant_id_eia,generator_id,report_year,fuel_group_code) %>%
  summarize(
    num_par_load_lines = sum(!(is.na(parasitic_load) | net_generation_mwh_final==0), na.rm = TRUE),
    parasitic_load = sum(ifelse(net_generation_mwh_final==0,NA,parasitic_load * fuel_frac), na.rm = TRUE),
    capacity_hist = sum(ifelse(is.na(parasitic_load) | net_generation_mwh_final==0,NA,
                               capacity_hist * fuel_frac / num_par_load_lines), na.rm = TRUE),
    num_hours = sum(month_hours),
    prime_mover_code_hist = prime_mover_code_hist,
    CF_para = (parasitic_load / (capacity_hist * num_hours))
  ) %>% ungroup() %>% distinct() %>%
  group_by(plant_id_eia,prime_mover_code_hist,report_year,fuel_group_code) %>%
  summarize(
    capacity_hist = sum(ifelse(is.na(parasitic_load),NA,
                               capacity_hist), na.rm = TRUE),
    parasitic_load = sum(parasitic_load, na.rm = TRUE)
  ) %>% ungroup() %>% distinct()

parasitic_load_ave <- parasitic_load_est %>%
  rename("fuel_group_code_ave" = "fuel_group_code") %>%
  group_by(plant_id_eia,prime_mover_code_hist,fuel_group_code_ave) %>%
  summarize(
    capacity_hist_ave = sum(ifelse(is.na(parasitic_load),NA,
                               capacity_hist), na.rm = TRUE),
    parasitic_load_ave = sum(parasitic_load, na.rm = TRUE)
  ) %>% ungroup() %>% distinct()

parasitic_load_agg <- parasitic_load_est %>%
  rename("fuel_group_code_agg" = "fuel_group_code") %>%
  group_by(prime_mover_code_hist,fuel_group_code_agg) %>%
  summarize(
    capacity_hist_agg = sum(ifelse(is.na(parasitic_load),NA,
                                   capacity_hist), na.rm = TRUE),
    parasitic_load_agg = sum(parasitic_load, na.rm = TRUE)
  ) %>% ungroup() %>% distinct()

  # Now we turn to estimating costs using FERC data matched to EIA and CEMS data at the plant-prime-fuel level

# First we bring in CPI and state-level wage data to normalize operating costs - and just CPI data for capex -
# to allow for national-level regression analyses.

CPIU_ranges<-xlsx_names("BBB Fossil Transition Analysis Inputs.xlsm") %>%
  subset(is_range == TRUE & hidden == FALSE, select = c(name,formula))
CPIU_range_formula<-CPIU_ranges %>%
  subset(name=="CPIU",select="formula") %>% as.character
CPIU<-read_excel("BBB Fossil Transition Analysis Inputs.xlsm",range = CPIU_range_formula)

#Using labor Quarterly Census of Employment and Wages data to normalize Fixed OpEx, from https://www.bls.gov/cew/downloadable-data-files.htm
#zipFiles <- list.files(path=".", pattern="*.zip", full.names=T, recursive=FALSE)

#Unzip each of the files
#for (f in zipFiles) {
#  year<-substring(f,3,6)
#  if (year<=2015) file = paste(year,'.annual.by_industry/',year,'.annual 2211 Power generation and supply.csv',sep="") else file = paste(year,'.annual.by_industry/',year,'.annual 2211 NAICS 2211 Power generation and supply.csv',sep="")
#  unzip(f,file,junkpaths=TRUE,exdir=".")
#}

#Construct summary data wage factors by state and year of state electric power sector wages relative to national average wages

years<-c("1994","1995","1996","1997","1998","1999","2000","2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015","2016","2017","2018", "2019", "2020")
for (year in years) {
  if (year<=2015) file = paste('wage_level_data/',year,'.annual 2211 Power generation and supply.csv',sep="") else file = paste('wage_level_data/',year,'.annual 2211 NAICS 2211 Power generation and supply.csv',sep="")
  filetemp <- read.csv(file)
  if(year>1994) wages_by_state<-rbind.data.frame(wages_by_state, filetemp) else wages_by_state <- filetemp
}
wages_by_state<-subset(wages_by_state,(agglvl_code==56))
wages_by_state$total_wages<-as.numeric(as.numeric(wages_by_state$avg_annual_pay) * as.numeric(wages_by_state$annual_avg_emplvl))
wages_by_state$area_fips<-as.numeric(wages_by_state$area_fips)
wages_summary_by_state <- wages_by_state[,c("area_fips","year","annual_avg_emplvl","total_wages")] %>%
  group_by(area_fips,year) %>%
  summarise(empl = sum(annual_avg_emplvl),tot_wages = sum(total_wages),.groups="drop")
wages_summary_by_state$wages <- ifelse(wages_summary_by_state$empl>0,wages_summary_by_state$tot_wages / wages_summary_by_state$empl,NA)
wages_summary_by_state <- wages_summary_by_state %>% group_by(year)
wages_summary_by_year <- wages_summary_by_state %>% summarise(ave_wages = mean(wages,na.rm = TRUE))
wages_summary_by_state <-wages_summary_by_state %>% merge(wages_summary_by_year,by = "year")
wages_summary_by_state$wage_scale = ifelse(is.na(wages_summary_by_state$wages),1,wages_summary_by_state$wages/wages_summary_by_state$ave_wages)
state_fips_map<-read.csv("State_FIPS_Match.csv")
keeps<-c("area_fips","State")
state_fips_map<-state_fips_map[keeps]
wages_summary_by_state <- wages_summary_by_state %>% merge(state_fips_map, by = "area_fips")
keeps <- c("year","wage_scale","State")
wage_factors <- wages_summary_by_state[keeps]

# Now we pull in historical FERC data that has been matched at the plant-prime-fuel level with
# EIA data in order estimate capex / opex / start and stop costs.

FERC_Data<-read_excel("860_FERC_matching_cost_regressions - values.xlsx",guess_max = 20000)
FERC_Data<-as.data.frame(unclass(FERC_Data),stringsAsFactors = TRUE, optional = TRUE)
FERC_Data <- merge(x=FERC_Data, y=wage_factors, by.x = c("report_year","State"),by.y = c("year","State"),all.x=TRUE)
FERC_Data$wage_scale = ifelse(is.na(FERC_Data$wage_scale),1,FERC_Data$wage_scale)

# Pull in parasitic load to generation gross CFs for regression analyses

FERC_Data <- FERC_Data %>%
  mutate(FERC_line = row_number()) %>%
  left_join(parasitic_load_est, by=c(
    "Plant" = "plant_id_eia",
    "Prime" = "prime_mover_code_hist",
    "report_year" = "report_year"
  )) %>% mutate(no_match = is.na(fuel_group_code)) %>%
  left_join(parasitic_load_ave %>% mutate(no_match=1), by=c(
    "Plant" = "plant_id_eia",
    "Prime" = "prime_mover_code_hist",
    "no_match" = "no_match"
  )) %>% mutate(no_match = is.na(fuel_group_code) & is.na(fuel_group_code_ave)) %>%
  left_join(parasitic_load_agg %>% mutate(no_match=1), by=c(
    "Prime" = "prime_mover_code_hist",
    "no_match" = "no_match"
  )) %>%
  mutate(
    fuel_group_code = ifelse(is.na(fuel_group_code),
                             ifelse(is.na(fuel_group_code_ave),fuel_group_code_agg,fuel_group_code_ave),
                             fuel_group_code),
    parasitic_load = ifelse(is.na(parasitic_load),
                            ifelse(is.na(parasitic_load_ave),parasitic_load_agg,parasitic_load_ave),
                            parasitic_load),
    capacity_hist = ifelse(is.na(capacity_hist),
                           ifelse(is.na(capacity_hist_ave),capacity_hist_agg,capacity_hist_ave),
                           capacity_hist),
    parasitic_load = case_when(
      fuel_group_code == "natural_gas" ~ natural_gas * parasitic_load,
      fuel_group_code == "coal" ~ coal * parasitic_load,
      fuel_group_code == "petroleum" ~ petroleum * parasitic_load,
      fuel_group_code == "petroleum_coke" ~ petroleum_coke * parasitic_load,
      fuel_group_code == "other_gas" ~ other_gas * parasitic_load,
      TRUE ~ 0 * parasitic_load
    ),
    capacity_hist = case_when(
      fuel_group_code == "natural_gas" ~ natural_gas * capacity_hist,
      fuel_group_code == "coal" ~ coal * capacity_hist,
      fuel_group_code == "petroleum" ~ petroleum * capacity_hist,
      fuel_group_code == "petroleum_coke" ~ petroleum_coke * capacity_hist,
      fuel_group_code == "other_gas" ~ other_gas * capacity_hist,
      TRUE ~ 0 * capacity_hist
    )
  ) %>%
  group_by(FERC_line) %>%
  mutate(
    parasitic_load = sum(parasitic_load, na.rm = TRUE),
    capacity_hist = sum(capacity_hist, na.rm = TRUE),
    parasitic_load_per_MW = ifelse(capacity_hist>0,parasitic_load/capacity_hist,0)
  ) %>% ungroup() %>%
  select(-c(
    parasitic_load,
    capacity_hist,
    fuel_group_code,
    parasitic_load_ave,
    capacity_hist_ave,
    fuel_group_code_ave,
    parasitic_load_agg,
    capacity_hist_agg,
    fuel_group_code_agg,
    no_match
  )) %>% distinct() %>%
  mutate(
    FERC_CF_net = FERC_CF,
    FERC_CF = FERC_CF + parasitic_load_per_MW / ifelse(report_year %% 4 ==0,8784,8760)
  )

# Construct fields for operating cost and capital cost regression analyses

## remove outliers

FERC_Data <- FERC_Data %>%
  mutate(opex_over_capex = opex_per_kW / capex_per_kW,
         outlier_flag = ifelse(opex_over_capex >=0.5 | opex_over_capex <=0.002 | FERC_CF>1.1, 1, 0)) %>%
  group_by(Plant, Prime) %>%
  mutate(outlier_count = sum(outlier_flag, na.rm=TRUE)) %>%
  ungroup()

## calculate historical median CF and capacity factor quartiles

FERC_Data<-FERC_Data %>%
  group_by(Plant, Prime) %>%
  mutate(median_CF = median(ifelse(FERC_CF<=1.1,FERC_CF,NA),na.rm=TRUE),
         median_OpEx = median(ifelse(FERC_CF<=1.05,real_opex,NA),na.rm=TRUE),
         high_median_CF = ifelse(median_CF>0.6,1,0),
         mid_median_CF = ifelse(median_CF>0.4 & median_CF<=0.6,1,0),
         low_median_CF = ifelse(median_CF<=0.2,1,0))

FERC_outliers<- FERC_Data %>% subset(outlier_flag==1)

# Bring in plant-prime level starts and stops from CEMS data to estimate start-up/stop costs

plant_prime_fuel_starts<-read_parquet("plant_prime_fuel_starts.parquet")
camd_eia_crosswalk_by_year<-read_parquet("camd_eia_crosswalk_by_year.parquet")

# Use CAMD EIA crosswalk to identify the number of units for each plant-prime-fuel in each reporting year

camd_eia_crosswalk_by_year<-camd_eia_crosswalk_by_year %>%
  mutate(
    Prime = ifelse(EIA_UNIT_TYPE=="CA" | EIA_UNIT_TYPE=="CT" | EIA_UNIT_TYPE=="CS","CC",EIA_UNIT_TYPE),
    Fuel = case_when(
      CAMD_FUEL_TYPE=="Coal" | CAMD_FUEL_TYPE=="Coal Refuse" ~ "coal",
      CAMD_FUEL_TYPE=="Pipeline Natural Gas" | CAMD_FUEL_TYPE=="Natural Gas" ~ "natural_gas",
      CAMD_FUEL_TYPE=="Diesel Oil" | CAMD_FUEL_TYPE=="Residual Oil" | CAMD_FUEL_TYPE=="Other Oil" ~ "petroleum",
      CAMD_FUEL_TYPE=="Petroleum Coke" ~ "petroleum_coke",
      CAMD_FUEL_TYPE=="Process Gas" | CAMD_FUEL_TYPE=="Other Gas" ~ "other_gas",
    ),
  ) %>%
  filter(capacity_in_year>0) %>%
  select(EIA_PLANT_ID,CAMD_UNIT_ID,Prime,Fuel,capacity_year) %>% distinct() %>%
  group_by(EIA_PLANT_ID,Prime,Fuel,capacity_year) %>%
  summarize(
    units = sum(capacity_year>0),
  ) %>% ungroup() %>% distinct()

# Pull in starts for each plant-prime-fuel, and normalize by the number of units

plant_prime_fuel_starts<- plant_prime_fuel_starts %>%
  mutate(
    report_year = year(report_date),
    Plant = plant_id_cems,
    Prime = prime_mover_code,
    "Fuel 1" = fuel_code
  ) %>%
  select(-"plant_id_cems",-"__index_level_0__",-"fuel_code",-"prime_mover_code",-"report_date") %>%
  left_join(camd_eia_crosswalk_by_year, by = c(
    "report_year" = "capacity_year",
    "Plant" = "EIA_PLANT_ID",
    "Prime" = "Prime",
    "Fuel 1" = "Fuel"
  )) %>%
  mutate(
    starts = ifelse(is.na(units), starts, starts / units)
  )

# Fill in starts using averages by CF quartile, prime mover, and state (with back-ups at higher levels of aggregation)

FERC_Data<-FERC_Data %>%
  left_join(plant_prime_fuel_starts, by = c(
    "report_year" = "report_year",
    "Plant" = "Plant",
    "Prime" = "Prime",
    "Fuel 1" = "Fuel 1"
  )) %>%
  group_by(State,Prime,high_median_CF,mid_median_CF,low_median_CF) %>%
  mutate(
    ave_starts = median(starts, na.rm = TRUE),
    starts = ifelse(is.na(starts),ave_starts,starts)
  ) %>% ungroup() %>%
  group_by(Prime,high_median_CF,mid_median_CF,low_median_CF) %>%
  mutate(
    ave_starts = median(starts, na.rm = TRUE),
    starts = ifelse(is.na(starts),ave_starts,starts)
  ) %>% ungroup() %>%
  group_by(Prime) %>%
  mutate(
    ave_starts = median(starts, na.rm = TRUE),
    starts = ifelse(is.na(starts),ave_starts,starts)
  ) %>% ungroup() %>%
  group_by(high_median_CF,mid_median_CF,low_median_CF) %>%
  mutate(
    ave_starts = median(starts, na.rm = TRUE),
    starts = ifelse(is.na(starts),ave_starts,starts)
  ) %>% select(-"ave_starts") %>% ungroup()

# Calculate the cumulative starts from the first year each plant-prime-fuel appeared in the FERC data, and
# calculate inflation adjusted average age of an observation in order to extract a maintenance capex estimate

max_year <- max(FERC_Data$report_year, na.rm = TRUE)
min_year <- min(FERC_Data$report_year, na.rm = TRUE)
FERC_Data <- FERC_Data %>% mutate(cum_starts = 0)
wage_factors <- wage_factors %>% mutate(age_obs_adj = 0)
for(rep_year in min_year:max_year) {
  FERC_Data <- FERC_Data %>%
    group_by(Plant,Prime) %>%
    mutate(
      cum_starts = ifelse(report_year==rep_year,sum(ifelse(report_year<=rep_year,starts,0), na.rm = TRUE),cum_starts)
    ) %>% ungroup()
  wage_factors <- wage_factors %>%
    group_by(State) %>%
    mutate(
      age_obs_adj = ifelse(year==rep_year,sum(ifelse(year>=rep_year,wage_scale,0), na.rm = TRUE),age_obs_adj)
    ) %>% ungroup()
}
FERC_Data <- FERC_Data %>%
  left_join(wage_factors %>% select(c(year,age_obs_adj,State)),by=c("report_year"="year","State"="State"))

# Now we turn to the process of regressing the FERC data to estimate capex and opex coefficients.

# Start with (mostly) nominal, unadjusted variables for total capital cost per kW regression. The reason for this
# is accounting - the original costs are reported as the sum of nominal costs in various years. However, as maintenance
# capex may reasonably be expected to be constant in annual, real, state-adjusted dollars, and as cumulative
# maintenance capex could represent some or substantially all the annual change in original costs, we do use a cumulative
# adjusted age of observation rather than the raw age of observation to extract a real maintence capex.

FERC_Data<-FERC_Data %>%
  mutate(
    capacity = operating_capacity_in_report_year * 1000,
    gas_fixed = natural_gas * capacity,
    oil_fixed = petroleum * capacity,
    other_fixed = other_gas * capacity,
    coal_fixed = coal * capacity,
    pollution_fixed = pollution_control_costs_per_kW * capacity,
    age_fixed = age_relative_to_average * capacity,
    CHP_fixed = CHP * capacity,
    duct_burners_fixed = duct_burners * capacity,
    bypass_hrsg_fixed = bypass_hrsg * capacity,
    gasification_fixed = gasification * capacity,
    ccs_fixed = ccs * capacity,
    fluidized_bed_fixed = fluidized_bed * capacity,
    pulverized_coal_fixed = pulverized_coal * capacity,
    stoker_fixed = stoker * capacity,
    other_comb_fixed = other_comb * capacity,
    subcritical_fixed = subcritical * capacity,
    supercritical_fixed = supercritical * capacity,
    ultrasupercritical_fixed = ultrasupercritical * capacity,
    coal_pollution_fixed = coal * pollution_control_costs_per_kW * capacity,
    coal_age_fixed = coal * age_relative_to_average * capacity,
    gas_pollution_fixed = natural_gas * pollution_control_costs_per_kW * capacity,
    gas_age_fixed = natural_gas * age_relative_to_average * capacity,
    oil_pollution_fixed = petroleum * pollution_control_costs_per_kW * capacity,
    oil_age_fixed = petroleum * age_relative_to_average * capacity,
    coal_CHP_fixed = CHP * coal * capacity,
    gas_CHP_fixed = CHP * natural_gas * capacity,
    oil_CHP_fixed = CHP * petroleum * capacity,
    other_CHP_fixed = CHP * other_gas * capacity,
    age_obs_fixed = capacity * age_obs_adj,
    cum_starts_age_obs_fixed = cum_starts * capacity * age_obs_adj,
    CHP_age_obs_fixed = CHP * capacity * age_obs_adj,
    duct_burners_age_obs_fixed = duct_burners * capacity * age_obs_adj,
    bypass_hrsg_age_obs_fixed = bypass_hrsg * capacity * age_obs_adj,
    gasification_age_obs_fixed = gasification * capacity * age_obs_adj,
    ccs_age_obs_fixed = ccs * capacity * age_obs_adj,
    fluidized_bed_age_obs_fixed = fluidized_bed * capacity * age_obs_adj,
    pulverized_coal_age_obs_fixed = pulverized_coal * capacity * age_obs_adj,
    stoker_age_obs_fixed = stoker * capacity * age_obs_adj,
    other_comb_age_obs_fixed = other_comb * capacity * age_obs_adj,
    subcritical_age_obs_fixed = subcritical * capacity * age_obs_adj,
    supercritical_age_obs_fixed = supercritical * capacity * age_obs_adj,
    ultrasupercritical_age_obs_fixed = ultrasupercritical * capacity * age_obs_adj,
    coal_age_obs_fixed = coal * capacity * age_obs_adj,
    gas_age_obs_fixed = natural_gas * capacity * age_obs_adj,
    oil_age_obs_fixed = petroleum * capacity * age_obs_adj,
    other_age_obs_fixed = other_gas * capacity * age_obs_adj
  )

# Now define state wage level adjusted, real cost variables for OpEx regression

FERC_Data<-FERC_Data %>%
  mutate(
    capacity_adj = wage_scale * capacity,
    starts_adj = starts * capacity_adj,
    gas_starts_adj = natural_gas * starts_adj,
    oil_starts_adj = petroleum * starts_adj,
    other_starts_adj = other_gas * starts_adj,
    coal_starts_adj = coal * starts_adj,
    gas_fixed_adj = natural_gas * capacity_adj,
    oil_fixed_adj = petroleum * capacity_adj,
    other_fixed_adj = other_gas * capacity_adj,
    coal_fixed_adj = coal * capacity_adj,
    high_median_CF_fixed_adj = high_median_CF * capacity_adj,
    mid_median_CF_fixed_adj = mid_median_CF * capacity_adj,
    low_median_CF_fixed_adj = low_median_CF * capacity_adj,
    median_CF_fixed_adj = median_CF * capacity_adj,
    pollution_fixed_adj = real_pollution_control_costs_per_kW * capacity_adj,
    age_fixed_adj = age_relative_to_average * capacity_adj,
    age_obs_fixed_adj = age_of_observation * capacity_adj,
    CHP_fixed_adj = CHP * capacity_adj,
    duct_burners_fixed_adj = duct_burners * capacity_adj,
    bypass_hrsg_fixed_adj = bypass_hrsg * capacity_adj,
    gasification_fixed_adj = gasification * capacity_adj,
    ccs_fixed_adj = ccs * capacity_adj,
    fluidized_bed_fixed_adj = fluidized_bed * capacity_adj,
    pulverized_coal_fixed_adj = pulverized_coal * capacity_adj,
    stoker_fixed_adj = stoker * capacity_adj,
    other_comb_fixed_adj = other_comb * capacity_adj,
    subcritical_fixed_adj = subcritical * capacity_adj,
    supercritical_fixed_adj = supercritical * capacity_adj,
    ultrasupercritical_fixed_adj = ultrasupercritical * capacity_adj,
    coal_pollution_fixed_adj = coal * real_pollution_control_costs_per_kW * capacity_adj,
    coal_age_fixed_adj = coal * age_relative_to_average * capacity_adj,
    gas_pollution_fixed_adj = natural_gas * real_pollution_control_costs_per_kW * capacity_adj,
    gas_age_fixed_adj = natural_gas * age_relative_to_average * capacity_adj,
    oil_pollution_fixed_adj = petroleum * real_pollution_control_costs_per_kW * capacity_adj,
    oil_age_fixed_adj = petroleum * age_relative_to_average * capacity_adj,
    gen = FERC_CF * operating_capacity_in_report_year * ifelse(report_year %% 4 ==0,8784,8760),
    gen_adj = wage_scale * gen,
    high_median_CF_adj = high_median_CF * gen_adj,
    mid_median_CF_adj = mid_median_CF * gen_adj,
    low_median_CF_adj = low_median_CF * gen_adj,
    median_CF_adj = median_CF * gen_adj,
    gas_variable_adj = natural_gas * gen_adj,
    oil_variable_adj = petroleum * gen_adj,
    other_variable_adj = other_gas * gen_adj,
    coal_variable_adj = coal * gen_adj,
    pollution_variable_adj = real_pollution_control_costs_per_kW * gen_adj,
    age_variable_adj = age_relative_to_average * gen_adj,
    age_obs_variable_adj = age_of_observation * gen_adj,
    CHP_variable_adj = CHP * gen_adj,
    duct_burners_variable_adj = duct_burners * gen_adj,
    bypass_hrsg_variable_adj = bypass_hrsg * gen_adj,
    gasification_variable_adj = gasification * gen_adj,
    ccs_variable_adj = ccs * gen_adj,
    fluidized_bed_variable_adj = fluidized_bed * gen_adj,
    pulverized_coal_variable_adj = pulverized_coal * gen_adj,
    stoker_variable_adj = stoker * gen_adj,
    other_comb_variable_adj = other_comb * gen_adj,
    subcritical_variable_adj = subcritical * gen_adj,
    supercritical_variable_adj = supercritical * gen_adj,
    ultrasupercritical_variable_adj = ultrasupercritical * gen_adj,
    coal_pollution_variable_adj = coal * real_pollution_control_costs_per_kW * gen_adj,
    coal_age_variable_adj = coal * age_relative_to_average * gen_adj,
    gas_pollution_variable_adj = natural_gas * real_pollution_control_costs_per_kW * gen_adj,
    gas_age_variable_adj = natural_gas * age_relative_to_average * gen_adj,
    oil_pollution_variable_adj = petroleum * real_pollution_control_costs_per_kW * gen_adj,
    oil_age_variable_adj = petroleum * age_relative_to_average * gen_adj
  )

#Average_Age_Data<-FERC_Data %>%
#  filter((natural_gas==1 | coal==1 | petroleum==1 | petroleum_coke==1 | other_gas==1))
#keeps<-c("Prime","natural_gas","coal","petroleum","petroleum_coke","other_gas","prime_fuels_average_age")
#Average_Age_Data<-Average_Age_Data[keeps]
#Average_Age_Data<-unique(Average_Age_Data)

#FERC_Data<-FERC_Data %>% ungroup()
#write.csv(FERC_Data, "FERC_Data.csv")

##OpEx Models

#ST Model - best option

STopexmodel<-lm(real_opex~
                  0
                #+capacity
                +capacity_adj
                +starts_adj
                +gas_starts_adj
                #+oil_starts_adj
                #+other_starts_adj
                +high_median_CF_fixed_adj
                +mid_median_CF_fixed_adj
                #+low_median_CF_fixed_adj
                +median_CF_fixed_adj
                #+coal_fixed_adj
                +gas_fixed_adj
                +oil_fixed_adj
                #+other_fixed_adj
                +pollution_fixed_adj
                #+age_fixed_adj
                #+age_obs_fixed_adj
                #+CHP_fixed_adj
                #+duct_burners_fixed_adj
                #+bypass_hrsg_fixed_adj
                #+gasification_fixed_adj
                #+ccs_fixed_adj
                #+fluidized_bed_fixed_adj
                +pulverized_coal_fixed_adj
                #+stoker_fixed_adj
                #+other_comb_fixed_adj
                #+subcritical_fixed_adj
                +supercritical_fixed_adj
                #+ultrasupercritical_fixed_adj
                #+coal_pollution_fixed_adj
                #+coal_age_fixed_adj
                +gas_pollution_fixed_adj
                +gas_age_fixed_adj
                #+oil_pollution_fixed_adj
                #+oil_age_fixed_adj
                #+gen
                +gen_adj
                +high_median_CF_adj
                +mid_median_CF_adj
                #+low_median_CF_adj
                +median_CF_adj
                #+coal_variable_adj
                #+gas_variable_adj
                #+oil_variable_adj
                #+other_variable_adj
                +pollution_variable_adj
                +age_variable_adj
                +age_obs_variable_adj
                +CHP_variable_adj
                #+duct_burners_variable_adj
                #+bypass_hrsg_variable_adj
                #+gasification_variable_adj
                #+ccs_variable_adj
                +fluidized_bed_variable_adj
                #+pulverized_coal_variable_adj
                #+stoker_variable_adj
                #+other_comb_variable_adj
                #+subcritical_variable_adj
                +supercritical_variable_adj
                #+ultrasupercritical_variable_adj
                #+coal_pollution_variable_adj
                #+coal_age_variable_adj
                #+gas_pollution_variable_adj
                +gas_age_variable_adj
                #+oil_pollution_variable_adj
                #+oil_age_variable_adj
                ,data=FERC_Data,subset=(ST==1 & FERC_CF<=1.1 & outlier_flag==0))
summary(STopexmodel)
#anova(STopexmodel,STopexmodel1)
#plot(fitted(STopexmodel), residuals(STopexmodel))
#plot(STopexmodel$model$real_opex, fitted(STopexmodel))
#abline(a=0, b=1)

## CC Model: Single CC does not account for differences in cost and operation between load-following and baseload CCs. However, eliminating
## extreme outlier plants and IGCCs in operating expenses seems to significantly improve its predictive power

CCopexmodel<-lm(real_opex~
                  0
                +capacity_adj
                +starts_adj
                #+oil_starts_adj
                #+other_starts_adj
                +high_median_CF_fixed_adj
                +mid_median_CF_fixed_adj
                +low_median_CF_fixed_adj
                +median_CF_fixed_adj
                #+coal_fixed_adj
                #+gas_fixed_adj
                #+oil_fixed_adj
                #+pollution_fixed_adj
                #+age_fixed_adj
                +CHP_fixed_adj
                +duct_burners_fixed_adj
                #+bypass_hrsg_fixed_adj
                #+gasification_fixed_adj
                #+ccs_fixed_adj
                #+age_obs_fixed_adj
                #+coal_pollution_fixed_adj
                #+coal_age_fixed_adj
                #+gas_pollution_fixed_adj
                #+gas_age_fixed_adj
                #+oil_pollution_fixed_adj
                +oil_age_fixed_adj
                +gen_adj
                +high_median_CF_adj
                +mid_median_CF_adj
                +low_median_CF_adj
                +median_CF_adj
                #+coal_variable_adj
                #+gas_variable_adj
                #+oil_variable_adj
                +pollution_variable_adj
                +age_variable_adj
                +age_obs_variable_adj
                #+CHP_variable_adj
                #+duct_burners_variable_adj
                #+bypass_hrsg_variable_adj
                #+gasification_variable_adj
                #+ccs_variable_adj
                #+coal_pollution_variable_adj
                #+coal_age_variable_adj
                #+gas_pollution_variable_adj
                #+gas_age_variable_adj
                #+oil_pollution_variable_adj
                #+oil_age_variable_adj
                ,data=FERC_Data,subset=(CC==1 & coal==0 & outlier_flag==0 & real_opex_percentile<=0.97 & real_opex_percentile>=0.03))
summary(CCopexmodel)
#anova(CCopexmodel,CCopexmodel1)
#plot(fitted(CCopexmodel), residuals(CCopexmodel))
#plot(CCopexmodel$model$real_opex, fitted(CCopexmodel))
#abline(a=0, b=1)

## GTopex

GTopexmodel<-lm(real_opex~
                  0
                +capacity_adj
                +starts_adj
                +oil_starts_adj
                #+other_starts_adj
                #+high_median_CF_fixed_adj
                +mid_median_CF_fixed_adj
                +low_median_CF_fixed_adj
                +median_CF_fixed_adj
                #+coal_fixed_adj
                #+gas_fixed_adj
                +oil_fixed_adj
                #+other_fixed_adj
                +age_fixed_adj
                +age_obs_fixed_adj
                #+CHP_fixed_adj
                #+coal_age_fixed_adj
                #+gas_age_fixed_adj
                +oil_age_fixed_adj
                +gen_adj
                +high_median_CF_adj
                +mid_median_CF_adj
                +low_median_CF_adj
                +median_CF_adj
                #+coal_variable_adj
                #+gas_variable_adj
                +oil_variable_adj
                #+other_variable_adj
                +age_variable_adj
                +age_obs_variable_adj
                #+CHP_variable_adj
                #+oil_age_variable_adj
                ,data=FERC_Data,subset=(GT==1 & outlier_flag==0 & real_opex_percentile<=0.97 & real_opex_percentile>=0.03))
summary(GTopexmodel)
#anova(GTopexmodel,GTopexmodel1)
#plot(fitted(GTopexmodel), residuals(GTopexmodel))
#plot(GTopexmodel$model$real_opex, fitted(GTopexmodel))
#abline(a=0, b=1)

## ICopex
ICopexmodel<-lm(real_opex~
                  0
                +capacity_adj
                +starts_adj
                +oil_starts_adj
                #+other_starts_adj
                #+high_median_CF_fixed_adj
                #+mid_median_CF_fixed_adj
                #+low_median_CF_fixed_adj
                +median_CF_fixed_adj
                #+coal_fixed_adj
                #+gas_fixed_adj
                +oil_fixed_adj
                +age_fixed_adj
                #+age_obs_fixed_adj
                #+coal_age_fixed_adj
                #+gas_age_fixed_adj
                #+oil_age_fixed_adj
                +gen_adj
                +high_median_CF_adj
                #+mid_median_CF_adj
                #+low_median_CF_adj
                +median_CF_adj
                #+coal_variable_adj
                #+gas_variable_adj
                #+oil_variable_adj
                #+age_variable_adj
                #+age_obs_variable_adj
                #+coal_age_variable_adj
                #+gas_age_variable_adj
                #+oil_age_variable_adj
                ,data=FERC_Data,subset=(IC==1 & outlier_flag==0 & real_opex_percentile<=0.99 & real_opex_percentile >=0.01))
summary(ICopexmodel)
#anova(ICopexmodel,ICopexmodel1)
#plot(fitted(ICopexmodel), residuals(ICopexmodel))
#plot(ICopexmodel$model$real_opex, fitted(ICopexmodel))
#abline(a=0, b=1)

## CapEx models

## STcapex - here, we do not adjust for state level wage differences as the total capital expenditures modeled
## reflect nominal investments over many years summed together

STcapexmodel<-lm(capex~
                   0
                 +capacity
                 #+coal_fixed
                 +gas_fixed
                 +oil_fixed
                 #+other_fixed
                 +pollution_fixed
                 +age_fixed
                 +CHP_fixed
                 #+duct_burners_fixed
                 #+bypass_hrsg_fixed
                 #+gasification_fixed
                 #+ccs_fixed
                 #+fluidized_bed_fixed
                 +pulverized_coal_fixed
                 #+stoker_fixed
                 +other_comb_fixed
                 #+subcritical_fixed
                 #+supercritical_fixed
                 +ultrasupercritical_fixed
                 #+coal_pollution_fixed
                 #+coal_age_fixed
                 +gas_pollution_fixed
                 +gas_age_fixed
                 #+oil_pollution_fixed
                 #+oil_age_fixed
                 +gas_CHP_fixed
                 +oil_CHP_fixed
                 #+other_CHP_fixed
                 +age_obs_fixed
                 +cum_starts_age_obs_fixed
                 +CHP_age_obs_fixed
                 #+coal_age_obs_fixed
                 +gas_age_obs_fixed
                 +oil_age_obs_fixed
                 #+duct_burners_age_obs_fixed
                 #+bypass_hrsg_age_obs_fixed
                 #+gasification_age_obs_fixed
                 #+ccs_age_obs_fixed
                 #+fluidized_bed_age_obs_fixed
                 +pulverized_coal_age_obs_fixed
                 #+stoker_age_obs_fixed
                 #+other_comb_age_obs_fixed
                 #+subcritical_age_obs_fixed
                 +supercritical_age_obs_fixed
                 #+ultrasupercritical_age_obs_fixed
                 ,data=FERC_Data,subset=(ST==1 & outlier_flag==0))
summary(STcapexmodel)
#anova(STcapexmodel,STcapexmodel1)
#plot(fitted(STcapexmodel), residuals(STcapexmodel))
#plot(STcapexmodel$model$capex, fitted(STcapexmodel))
#abline(a=0, b=1)

## CCcapex - Here, we remove coal IGCC assets from the data to focus on the CCGTs

CCcapexmodel<-lm(capex~
                   0
                 +capacity
                 #+coal_fixed
                 #+gas_fixed
                 #+oil_fixed
                 #+pollution_fixed
                 +age_fixed
                 +CHP_fixed
                 #+gasification_fixed
                 #+ccs_fixed
                 #+duct_burners_fixed
                 +bypass_hrsg_fixed
                 #+coal_pollution_fixed
                 #+coal_age_fixed
                 #+gas_pollution_fixed
                 #+gas_age_fixed
                 #+oil_pollution_fixed
                 #+oil_age_fixed
                 #+age_obs_fixed
                 +cum_starts_age_obs_fixed
                 #+CHP_age_obs_fixed
                 #+coal_age_obs_fixed
                 #+gas_age_obs_fixed
                 #+oil_age_obs_fixed
                 +duct_burners_age_obs_fixed
                 +bypass_hrsg_age_obs_fixed
                 #+gasification_age_obs_fixed
                 #+ccs_age_obs_fixed
                 ,data=FERC_Data,subset=(CC==1 & coal==0 & outlier_flag==0))
summary(CCcapexmodel)
#plot(fitted(CCcapexmodel), residuals(CCcapexmodel))
#plot(CCcapexmodel$model$capex, fitted(CCcapexmodel))
#abline(a=0, b=1)

## GTcapex

GTcapexmodel<-lm(capex~
                   0
                 +capacity
                 #+coal_fixed
                 #+gas_fixed
                 +oil_fixed
                 #+pollution_fixed
                 +age_fixed
                 +CHP_fixed
                 #+coal_pollution_fixed
                 #+coal_age_fixed
                 #+gas_pollution_fixed
                 #+gas_age_fixed
                 #+oil_pollution_fixed
                 +oil_age_fixed
                 +age_obs_fixed
                 +cum_starts_age_obs_fixed
                 +CHP_age_obs_fixed
                 #+coal_age_obs_fixed
                 #+gas_age_obs_fixed
                 +oil_age_obs_fixed
                 ,data=FERC_Data,subset=(GT==1 & outlier_flag==0))
summary(GTcapexmodel)
#plot(fitted(GTcapexmodel), residuals(GTcapexmodel))
#plot(GTcapexmodel$model$capex, fitted(GTcapexmodel))
#abline(a=0, b=1)

## ICcapex

ICcapexmodel<-lm(capex~
                   0
                 +capacity
                 #+coal_fixed
                 +gas_fixed
                 #+oil_fixed
                 #+pollution_fixed
                 +age_fixed
                 #+coal_pollution_fixed
                 #+coal_age_fixed
                 #+gas_pollution_fixed
                 +gas_age_fixed
                 #+oil_pollution_fixed
                 #+oil_age_fixed
                 #+age_obs_fixed
                 +cum_starts_age_obs_fixed
                 #+coal_age_obs_fixed
                 #+gas_age_obs_fixed
                 #+oil_age_obs_fixed
                 ,data=FERC_Data,subset=(IC==1 & outlier_flag==0))
summary(ICcapexmodel)
#anova(ICcapexmodel,ICcapexmodel1)
#plot(fitted(ICcapexmodel), residuals(ICcapexmodel))
#plot(ICcapexmodel$model$capex, fitted(ICcapexmodel))
#abline(a=0, b=1)


## get reg variables
#capex
STcapex_variables <- tidy(STcapexmodel)
STcapex_variables <- STcapex_variables %>%
  mutate(Prime = "ST")
CCcapex_variables <- tidy(CCcapexmodel)
CCcapex_variables <- CCcapex_variables %>%
  mutate(Prime = "CC")
GTcapex_variables <- tidy(GTcapexmodel)
GTcapex_variables <- GTcapex_variables %>%
  mutate(Prime = "GT")
ICcapex_variables <- tidy(ICcapexmodel)
ICcapex_variables <- ICcapex_variables %>%
  mutate(Prime = "IC")

reg_variables_capex <- rbind(STcapex_variables, CCcapex_variables, GTcapex_variables, ICcapex_variables)
drops <- c("std.error", "statistic","p.value")
reg_variables_capex <- reg_variables_capex[, !names(reg_variables_capex) %in% drops]
reg_variables_capex <- reg_variables_capex %>%
  relocate(Prime)
unique(reg_variables_capex$term)

reg_variables_capex <- reg_variables_capex %>%
  pivot_wider(names_from = term, values_from = estimate)

#opex
STopex_variables <- tidy(STopexmodel)
STopex_variables <- STopex_variables %>%
  mutate(Prime = "ST")
CCopex_variables <- tidy(CCopexmodel)
CCopex_variables <- CCopex_variables %>%
  mutate(Prime = "CC")
GTopex_variables <- tidy(GTopexmodel)
GTopex_variables <- GTopex_variables %>%
  mutate(Prime = "GT")
ICopex_variables <- tidy(ICopexmodel)
ICopex_variables <- ICopex_variables %>%
  mutate(Prime = "IC")

reg_variables_opex <- rbind(STopex_variables, CCopex_variables, GTopex_variables, ICopex_variables)
drops <- c("std.error", "statistic","p.value")
reg_variables_opex <- reg_variables_opex[, !names(reg_variables_opex) %in% drops]
reg_variables_opex <- reg_variables_opex %>%
  relocate(Prime)
unique(reg_variables_opex$term)

reg_variables_opex <- reg_variables_opex %>%
  pivot_wider(names_from = term, values_from = estimate)

# merge reg_variables and turn all NAs into 0s for predictions.
reg_variables <- merge(reg_variables_capex,reg_variables_opex, by="Prime")
cols<-colnames(reg_variables)
reg_variables[cols]<-lapply(reg_variables[cols], function(x) ifelse(is.na(x),0,x))

write_parquet(reg_variables,"FERC_cost_regressions_coefficients.parquet")

# Now, we use the opex and capex coefficiencts to estimate generator-level (actually, unit-level in the case of CCs)
# opex and capex for all EIA-860 reporting fossil assets, making use of EIA-860, EIA-923, and CEMS data to
# characterize the generators.

# First, we read in the fuel group and emissions map we have generated from EPA data and EIA documentation

fuel_group_and_emissions_map <- read_excel("fuel_group_and_emissions_map.xlsx")

fuel_map <- fuel_group_and_emissions_map %>% select(c(energy_source_code,fuel_group_code))

# Now, pull in current unit level data from EIA-860M and process for use to create both historical generation / fuel use and
# a counterfactual data set for use in the Patio model

all_years <- c(2008:2020)
all_months <- c(1:12)

unit_level_data <- read_parquet("../patio_data/unit_level_costs_with_flag.parquet")

unit_level_data <- unit_level_data %>%
  rename(
    "plant_id_eia" = "Plant_ID",
    "prime_mover_code" = "Prime_with_CCs",
    "generator_id" = "Generator_ID"
  )

# Create plant level grouping file for aggregation

plant_grouping <- unit_level_data %>%
  select(c(
    plant_id_eia,
    State,
    Balancing_Authority_Code,
    Latitude,
    Longitude
  )) %>% distinct()

# Next, we create unit level data for all years and months for each operating and retired asset

unit_level_data <- unit_level_data %>%
  left_join(all_years, by = character(), copy = TRUE) %>% rename(report_year = y) %>%
  left_join(all_months, by = character(), copy = TRUE) %>% rename(report_month = y) %>%
  mutate(month_hours = unname(24*days_in_month(ymd(report_year*10000+report_month*100+1))))

# Filter unit level data for all years and months to create a list of plant and generator ids of
# assets operating in each month and year for historical analysis

units_present_hist <- unit_level_data %>%
  filter(
    ((Operating_Year < report_year) | ((Operating_Year==report_year) & (Operating_Month <= report_month))) &
      (is.na(Retirement_Year) |
         (Retirement_Year > report_year) |
         ((Retirement_Year==report_year ) & (Retirement_Month > report_month))) &
      (is.na(Planned_Retirement_Year) |
         (Planned_Retirement_Year > report_year) |
         ((Planned_Retirement_Year==report_year ) & (Planned_Retirement_Month > report_month)))) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_year,
    report_month,
    month_hours
  ))

# Create unit level data set for counterfactual analysis, which includes assets operating today for all years
# regardless of start of operations as well as all retired assets until their retirement date.

unit_level_data_cf <- unit_level_data %>%
  filter(
    (is.na(Retirement_Year) |
       (Retirement_Year > report_year) |
       ((Retirement_Year==report_year ) & (Retirement_Month > report_month))) &
      (is.na(Planned_Retirement_Year) |
         (Planned_Retirement_Year > report_year) |
         ((Planned_Retirement_Year==report_year ) & (Planned_Retirement_Month > report_month))))

# Extract just key generator state / BAC identifier fields of all units present in the counterfactual case

units_present_cf <- unit_level_data_cf %>%
  select(c(
    plant_id_eia,
    generator_id,
    prime_mover_code,
    operational_capacity_in_report_year,
    State,
    Balancing_Authority_Code,
    report_year,
    report_month,
    month_hours
  )) %>% rename("prime_mover_code_hist" = "prime_mover_code","capacity_hist"="operational_capacity_in_report_year")

# Extract just the ntile data by generator to be used to map in estimated CFs

current_ntiles <- historic_860_923_data %>%
  filter(!is.na(fuel_group_code) & report_year==2020) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_month,
    fuel_group_code,
    cap_tile,
    lat_tile,
    long_tile,
  )) %>% distinct()

# Extract table with just historical average CF / starts data by state, BAC, ntiles to fill in counterfactual years

CF_starts_table <- historic_860_923_data %>%
  filter(!is.na(fuel_group_code)) %>%
  select(c(
    report_year,
    report_month,
    prime_mover_code_hist,
    fuel_group_code,
    State,
    Balancing_Authority_Code,
    cap_tile,
    lat_tile,
    long_tile,
    CF_net_ave,
    CF_gross_ave,
    gen_starts_ave,
    fuel_starts_ave
  )) %>% distinct()

# Pull together current unit CF, fuel fracs, and heat rates to develop counterfactual operations

current_unit_CFs_fuel_fracs_heat_rates <- historic_860_923_data %>%
  filter(!is.na(fuel_group_code) & report_year>=2018) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_month,
    report_year,
    fuel_group_code,
    month_hours,
    capacity_hist,
    unit_gen_code,
    net_generation_mwh_final,
    gross_gen_final,
    mmbtu_final,
    mmbtu_fg_final,
    mmbtu_per_mwh_ave,
    co2_fg,
    co2e_fg
  )) %>% distinct() %>%
  group_by(plant_id_eia,generator_id,report_month,fuel_group_code,unit_gen_code) %>%
  summarize(
    month_hours = sum(month_hours, na.rm = TRUE)/sum(!is.na(report_year)),
    capacity_hist = max(capacity_hist, na.rm = TRUE),
    net_generation_mwh_final = sum(net_generation_mwh_final, na.rm = TRUE)/sum(!is.na(report_year)),
    gross_gen_final = sum(ifelse(mmbtu_final>0,gross_gen_final,NA), na.rm = TRUE)/
      sum(!is.na(gross_gen_final) & !is.na(mmbtu_final) & (mmbtu_final>0)),
    mmbtu_final = sum(mmbtu_final, na.rm = TRUE)/sum(!is.na(mmbtu_final)),
    mmbtu_fg_final = sum(mmbtu_fg_final, na.rm = TRUE)/sum(!is.na(mmbtu_fg_final)),
    co2_fg = sum(ifelse(mmbtu_fg_final>0,co2_fg,NA), na.rm = TRUE)/sum(!is.na(mmbtu_fg_final)),
    co2e_fg = sum(ifelse(mmbtu_fg_final>0,co2e_fg,NA), na.rm = TRUE)/sum(!is.na(mmbtu_fg_final)),
    mmbtu_per_mwh_ave = mmbtu_per_mwh_ave
  ) %>% ungroup() %>% distinct() %>%
  mutate(
    fuel_frac = ifelse(mmbtu_final>0, mmbtu_fg_final / mmbtu_final,NA),
    CF_gross = ifelse(capacity_hist>0, gross_gen_final / (capacity_hist * month_hours),NA),
    CF_net = ifelse(capacity_hist>0, net_generation_mwh_final / (capacity_hist * month_hours),NA),
    co2_per_mmbtu_fg = ifelse(mmbtu_fg_final>0, co2_fg / mmbtu_fg_final,NA),
    co2e_per_mmbtu_fg = ifelse(mmbtu_fg_final>0, co2e_fg / mmbtu_fg_final,NA),
    mmbtu_per_gross_gen = ifelse(gross_gen_final>0, mmbtu_final / gross_gen_final, mmbtu_per_mwh_ave),
    mmbtu_per_net_gen = ifelse(net_generation_mwh_final>0, mmbtu_final / net_generation_mwh_final,NA)
  ) %>% select(-c(
    month_hours,
    capacity_hist,
    mmbtu_per_mwh_ave,
    net_generation_mwh_final,
    gross_gen_final,
    mmbtu_final,
    mmbtu_fg_final,
    co2_fg,
    co2e_fg,
  )) %>% distinct()

# For all the units present, generate the counterfactual generation and fuel use by first pulling in the current
# unit CFs, fuel fracs, and heat rates - and applying them to all counterfactual historical years (this allows each
# such historical year to be transformed into a correlated weather year for current generators), then using the
# ntiles to map in historical average CF / starts, and then fuel costs.

generation_fuel_table_cf <- units_present_cf %>%
  left_join(current_unit_CFs_fuel_fracs_heat_rates, by=c(
    "plant_id_eia" = "plant_id_eia",
    "report_month" = "report_month",
    "generator_id" = "generator_id"
  )) %>%
  left_join(current_ntiles,by=c(
    "plant_id_eia" = "plant_id_eia",
    "report_month" = "report_month",
    "generator_id" = "generator_id",
    "fuel_group_code" = "fuel_group_code"
  )) %>%
  left_join(CF_starts_table,by=c(
    "report_year"="report_year",
    "report_month"="report_month",
    "prime_mover_code_hist"="prime_mover_code_hist",
    "fuel_group_code"="fuel_group_code",
    "State"="State",
    "Balancing_Authority_Code"="Balancing_Authority_Code",
    "cap_tile"="cap_tile",
    "lat_tile"="lat_tile",
    "long_tile"="long_tile"
  )) %>%
  mutate(
    CF_net = ifelse(is.na(CF_net),CF_net_ave,CF_net),
    CF_gross = ifelse(is.na(CF_gross),CF_gross_ave,CF_gross),
    net_generation_mwh_final = CF_net * month_hours * capacity_hist,
    gross_gen_final = CF_gross * month_hours * capacity_hist,
    mmbtu_final = mmbtu_per_gross_gen * gross_gen_final,
    mmbtu_fg_final = fuel_frac * mmbtu_final,
    co2_fg = co2_per_mmbtu_fg * mmbtu_fg_final,
    co2e_fg = co2e_per_mmbtu_fg * mmbtu_fg_final
  ) %>%
  left_join(final_fuel_costs, by = c(
    "plant_id_eia" = "plant_id_eia",
    "prime_mover_code_hist" = "prime_mover_code",
    "report_year" = "report_year",
    "report_month" = "report_month",
    "fuel_group_code" = "fuel_group_code"
  ))

# Now, map in the most recent actual FERC operating costs and capex if reported.

generation_fuel_table_cf <- generation_fuel_table_cf %>%
  left_join(FERC_Data %>% filter(report_year == 2020) %>% select(c(
    Plant,
    Prime,
    capex_per_kW,
    real_opex_per_kW,
    natural_gas,
    coal,
    petroleum,
    petroleum_coke,
    other_gas
  )) %>%
    pivot_longer(cols = c("natural_gas","coal","petroleum","petroleum_coke","other_gas"),
                 names_to = "fuel_group_code",
                 values_to = "FERC_fuel_frac",
                 values_drop_na = TRUE) %>%
    filter(FERC_fuel_frac>0), by = c(
    "prime_mover_code_hist" = "Prime",
    "plant_id_eia" = "Plant",
    "fuel_group_code" = "fuel_group_code"
  )) %>% distinct() %>%
  group_by(plant_id_eia,generator_id) %>%
  mutate(
    FERC_wt = ifelse(is.na(FERC_fuel_frac),0,FERC_fuel_frac * fuel_frac)/
      sum(ifelse(is.na(FERC_fuel_frac),0,FERC_fuel_frac * fuel_frac),na.rm=TRUE),
    capex_per_kW_sum = sum(FERC_wt * capex_per_kW, na.rm = TRUE),
    real_opex_per_kW_sum = sum(FERC_wt * real_opex_per_kW, na.rm = TRUE)
  ) %>%
  select(-c(FERC_fuel_frac,FERC_wt,capex_per_kW,real_opex_per_kW)) %>% distinct() %>%
  rename("capex_per_kW" = "capex_per_kW_sum","real_opex_per_kW" = "real_opex_per_kW_sum")

# Write out this relatively complete dataset of counterfactuals, with CF and reported OpEx at a monthly
# level

write_parquet(generation_fuel_table_cf,"generation_fuel_table_cf.parquet")

# Construct a wider table for mapping in FERC regression coefficients for CapEx / OpEx

generation_fuel_table <- generation_fuel_table_cf %>%
  filter(!is.na(fuel_group_code)) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_year,
    report_month,
    prime_mover_code_hist,
    fuel_group_code,
    State,
    Balancing_Authority_Code,
    cap_tile,
    lat_tile,
    long_tile,
    month_hours,
    capacity_hist,
    unit_gen_code,
    net_generation_mwh_final,
    gross_gen_final,
    mmbtu_final,
    mmbtu_fg_final,
    fuel_frac,
    co2_fg,
    co2e_fg,
    final_fuel_cost_per_mmbtu
  )) %>% distinct() %>%
  group_by(plant_id_eia,generator_id,report_year,report_month) %>%
  mutate(
    final_fuel_cost = final_fuel_cost_per_mmbtu * mmbtu_final * fuel_frac,
    final_fuel_cost_tot = sum(final_fuel_cost),
    co2_tot = sum(co2_fg),
    co2e_tot = sum(co2e_fg)
  ) %>% ungroup() %>%
  rename("co2" = "co2_fg", "co2e" = "co2e_fg") %>%
  pivot_wider(names_from = fuel_group_code,
              names_glue = "{.value}_{fuel_group_code}",
              values_from = c(mmbtu_fg_final,fuel_frac,co2,co2e,final_fuel_cost),
              values_fn = list(mmbtu_fg_final = max,fuel_frac = max, co2 = max, co2e = max, final_fuel_cost = max),
              values_fill = 0
              )

# Annualize the data and calculate key metrics used in the remainder of the analysis

generation_fuel_table_annual <- generation_fuel_table %>%
  group_by(plant_id_eia, generator_id, report_year) %>%
  summarize(
    total_fuel_mmbtu = sum(ifelse(is.na(mmbtu_final),0,mmbtu_final)),
    total_fossil_mwh = sum(ifelse(is.na(net_generation_mwh_final),0,net_generation_mwh_final)),
    total_fossil_gross_mwh = sum(ifelse(is.na(gross_gen_final),0,gross_gen_final)),
    total_carbon = sum(ifelse(is.na(co2e_tot),0,co2e_tot)),
    total_CO2_emissions = sum(ifelse(is.na(co2_tot),0,co2_tot)),
    final_fuel_cost = sum(ifelse(is.na(final_fuel_cost_tot),0,final_fuel_cost_tot)),
    fuel_frac_natural_gas = ifelse(total_fuel_mmbtu>0,
                                   sum(ifelse(is.na(mmbtu_fg_final_natural_gas),0,
                                              mmbtu_fg_final_natural_gas))/total_fuel_mmbtu,0),
    fuel_frac_coal = ifelse(total_fuel_mmbtu>0,
                            sum(ifelse(is.na(mmbtu_fg_final_coal),0,
                                       mmbtu_fg_final_coal))/total_fuel_mmbtu,0),
    fuel_frac_petroleum = ifelse(total_fuel_mmbtu>0,
                                 sum(ifelse(is.na(mmbtu_fg_final_petroleum),0,
                                            mmbtu_fg_final_petroleum))/total_fuel_mmbtu,0),
    fuel_frac_petroleum_coke = ifelse(total_fuel_mmbtu>0,
                                      sum(ifelse(is.na(mmbtu_fg_final_petroleum_coke),0,
                                                 mmbtu_fg_final_petroleum_coke))/total_fuel_mmbtu,0),
    fuel_frac_other = ifelse(total_fuel_mmbtu>0,
                             sum(ifelse(is.na(mmbtu_fg_final_other),0,
                                        mmbtu_fg_final_other))/total_fuel_mmbtu,0),
    fuel_frac_other_gas = ifelse(total_fuel_mmbtu>0,
                                 sum(ifelse(is.na(mmbtu_fg_final_other_gas),0,
                                            mmbtu_fg_final_other_gas))/total_fuel_mmbtu,0),
    plant_id_eia = plant_id_eia,
    generator_id = generator_id,
    capacity_hist = capacity_hist,
#    prime_mover_code = prime_mover_code_hist,
#    fuel_code = fuel_group_code,
#    utility_id_eia = utility_id_eia,
    report_year = report_year
#    state = State
  ) %>% ungroup() %>% distinct()

# Pull in the annualized generation fuel and cost counterfactual data into the original unit level data
# to pull in the FERC regression coefficients to create patio cost inputs table, including estimated CapEx / OpEx

python_inputs_data <- unit_level_data_cf %>%
  select(-c(report_month,month_hours)) %>% distinct() %>%
  select(-starts_with("rmi_")) %>%
  select(-starts_with("prime_mover_code_")) %>%
  select(-starts_with("fuel_code_")) %>%
  select(-starts_with("fuel_group_")) %>%
  select(-starts_with("energy_source_code_")) %>%
  left_join(generation_fuel_table_annual, by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year"
  ))

python_inputs_data<-python_inputs_data %>%
  group_by(plant_id_eia, generator_id) %>%
  mutate(
    fossil_CF = ifelse(capacity_hist>0,total_fossil_gross_mwh / (capacity_hist * ifelse(report_year %% 4 ==0,8784,8760)),0),
    median_CF = median(ifelse(fossil_CF<=1,fossil_CF,1),na.rm=TRUE),
    high_median_CF = ifelse(median_CF>0.6,1,0),
    mid_median_CF = ifelse(median_CF>0.4 & median_CF<=0.6,1,0),
    low_median_CF = ifelse(median_CF<=0.2,1,0),
    pollution_control_costs_per_kW = ifelse(is.na(pollution_control_costs_per_kW),0,pollution_control_costs_per_kW)
  ) %>% ungroup()

#merge python_inputs_data and reg_variables
python_inputs_data <- python_inputs_data %>%
  left_join(reg_variables, by = c(
    "prime_mover_code" = "Prime"
  ))

#add state wage adjustment factors

python_inputs_data <- python_inputs_data %>%
  left_join(wage_factors, by = c(
    "State" = "State",
    "report_year" = "year"
  ))

# Calculate estimated capex per kW and annual maintenance capex costs, in nominal dollars - note that we
# break out the contribution to annual maintenance capex correlated with cumulative starts/stops to be
# dynamically calculated based on patio model starts and stops

python_inputs_data <- python_inputs_data %>%
  mutate(
    real_maint_capex_per_kW_no_cum_starts_est =
      -(age_obs_fixed +
          (gas_age_obs_fixed * fuel_frac_natural_gas) +
          (oil_age_obs_fixed * fuel_frac_petroleum) +
          (CHP_age_obs_fixed * chp) +
          (pulverized_coal_age_obs_fixed * pulverized_coal) +
          (supercritical_age_obs_fixed * supercritical) +
          (duct_burners_age_obs_fixed * duct_burners) +
          (bypass_hrsg_age_obs_fixed * bypass_hrsg)),
    real_maint_capex_per_kW_cum_starts_coeff = -cum_starts_age_obs_fixed,
    capex_per_kW_no_cum_starts_est =
      capacity +
      (gas_fixed * fuel_frac_natural_gas) +
      (oil_fixed * fuel_frac_petroleum) +
      (pollution_fixed * pollution_control_costs_per_kW) +
      (age_fixed * diff_age_and_avg_plant_prime_fuel) +
      (CHP_fixed * chp) +
      (pulverized_coal_fixed * pulverized_coal) +
      (other_comb_fixed * other_comb) +
      (ultrasupercritical_fixed * ultrasupercritical) +
      (gas_pollution_fixed * fuel_frac_natural_gas * pollution_control_costs_per_kW) +
      (gas_age_fixed * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel) +
      (oil_age_fixed * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel) +
      (gas_CHP_fixed * fuel_frac_natural_gas * chp) +
      (oil_CHP_fixed * fuel_frac_petroleum * chp) +
      #(duct_burners_fixed * duct_burners) +
      (bypass_hrsg_fixed * bypass_hrsg)
  )

# Calculate estimated real fixed costs opex as well as the start/stop costs. Note that we again break out
# the impact on real fixed opex of start/stops. Further, we break out the coefficient dependent on the age
# of observation for use only in historic cost estimation.

python_inputs_data <- python_inputs_data %>%
  mutate(real_fixed_opex_per_kW_no_starts_est = (
    (capacity_adj * wage_scale) +
      (median_CF_fixed_adj * median_CF * wage_scale ) +
      (high_median_CF_fixed_adj * high_median_CF * wage_scale ) +
      (mid_median_CF_fixed_adj * mid_median_CF * wage_scale ) +
      (low_median_CF_fixed_adj * low_median_CF * wage_scale ) +
      (gas_fixed_adj * fuel_frac_natural_gas * wage_scale) +
      (oil_fixed_adj * fuel_frac_petroleum * wage_scale) +
      (CHP_fixed_adj * chp * wage_scale) +
      (pollution_fixed_adj * pollution_control_costs_per_kW * wage_scale) +
      (pulverized_coal_fixed_adj * pulverized_coal * wage_scale) +
      (duct_burners_fixed_adj * duct_burners * wage_scale) +
      (supercritical_fixed_adj * supercritical * wage_scale) +
      (age_fixed_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (oil_age_fixed_adj * diff_age_and_avg_plant_prime_fuel * fuel_frac_petroleum * wage_scale) +
      (gas_age_fixed_adj * diff_age_and_avg_plant_prime_fuel * fuel_frac_natural_gas * wage_scale) +
      (gas_pollution_fixed_adj * fuel_frac_natural_gas * pollution_control_costs_per_kW * wage_scale)),
    real_opex_per_kW_start = starts_adj +
      (gas_starts_adj * fuel_frac_natural_gas) +
      (oil_starts_adj * fuel_frac_petroleum),
    real_fixed_opex_per_kW_age_coeff = age_obs_fixed_adj * wage_scale
    )

# Calculate estimated real variable costs opex, again breaking out coefficients dependent on age of observation
# only for historical cost estimation

python_inputs_data <- python_inputs_data %>%
  mutate(real_variable_opex_per_MWh_est = (
    (gen_adj * wage_scale) +
      (median_CF_adj * median_CF * wage_scale ) +
      (high_median_CF_adj * high_median_CF * wage_scale ) +
      (mid_median_CF_adj * mid_median_CF * wage_scale ) +
      (low_median_CF_adj * low_median_CF * wage_scale ) +
      #(gas_variable_adj * fuel_frac_natural_gas * wage_scale) +
      (oil_variable_adj * fuel_frac_petroleum * wage_scale) +
      (pollution_variable_adj * pollution_control_costs_per_kW * wage_scale) +
      (age_variable_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (fluidized_bed_variable_adj * fluidized_bed * wage_scale) +
      (CHP_variable_adj * chp * wage_scale) +
      (supercritical_variable_adj * supercritical * wage_scale) +
      #(duct_burners_variable_adj * duct_burners * wage_scale) +
      #(oil_age_variable_adj * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (gas_age_variable_adj * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel * wage_scale)),
    real_variable_opex_per_MWh_age_coeff = (age_obs_variable_adj * wage_scale)
    )


python_inputs_data <- python_inputs_data %>%
  left_join(select(CPIU,c(Year,Inflation_Factor_2020)), by = c("report_year" = "Year")) %>%
  mutate(
    #real_fixed_opex_per_kW = ifelse(is.na(real_opex_per_kW),real_est_fixed_opex_per_kW,real_opex_per_kW) -
    #  real_est_variable_opex_per_MWh * ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020),
    #real_est_OpEx = capacity_adjusted * 1000 * ifelse(is.na(real_opex_per_kW),real_est_fixed_opex_per_kW,real_opex_per_kW) +
    #       real_est_variable_opex_per_MWh * (total_fossil_mwh -ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020)),
    real_fuel_costs = final_fuel_cost / Inflation_Factor_2020,
    real_fuel_costs_per_MWh = real_fuel_costs / total_fossil_gross_mwh,
    #real_marginal_costs_per_MWh = real_fuel_costs_per_MWh + real_est_variable_opex_per_MWh,
    #real_operating_costs = real_est_OpEx + real_fuel_costs,
    #real_operating_costs_per_kW = real_operating_costs / (capacity_adjusted *1000),
    #real_operating_costs_per_MWh = real_operating_costs / total_fossil_mwh,
    emissions_intensity = total_carbon *1000000 / total_fossil_gross_mwh
  )

#average_fossil_cost_BA <- python_inputs_data %>%
#  select(c(Balancing_Authority_Code,total_fossil_mwh,report_year,real_operating_costs,total_carbon,real_operating_costs_per_MWh,emissions_intensity)) %>%
#  group_by(Balancing_Authority_Code,report_year) %>%
#  summarize(BA_OpEx = sum(real_operating_costs,na.rm = TRUE),
#            BA_gen = sum(total_fossil_mwh,na.rm = TRUE),
#            BA_emissions = sum(total_carbon,na.rm = TRUE)) %>%
#  ungroup %>%
#  unique %>%
#  mutate(average_BA_cost_per_mwh = BA_OpEx / BA_gen,
#         average_BA_emissions_intensity = BA_emissions / BA_gen)

#python_inputs_data <- python_inputs_data %>%
#  left_join(average_fossil_cost_BA,
#            by = c("Balancing_Authority_Code" = "Balancing_Authority_Code",
#                   "report_year" = "report_year"))

# Jettison intermediate data fields for subsequent analysis

keeps <- c("plant_id_eia",
           "prime_mover_code",
           "fuel_code",
           "report_year",
           "state",
           "Plant.Name",
           "Sector",
           "plant_prime_fuel_1",
           "Balancing_Authority_Code",
           "capacity_in_report_year",
           "capacity_adjusted",
           "capacity_adjustment",
           "total_fossil_mwh",
           "total_fuel_mmbtu",
           "total_carbon",
           "total_CO2_emissions",
           "final_fuel_cost",
           "fossil_CF",
           "median_CF",
           "capacity_in_2020",
           "capacity_in_2025",
           "capacity_in_2030",
           "capacity_in_2035",
           "average_BA_cost_per_mwh",
           "average_BA_emissions_intensity",
           "BA_OpEx",
           "BA_gen",
           "BA_emissions",
           "capex_per_kW",
           "real_opex_per_kW",
           "capex_per_kW_est",
           "maint_capex_per_kW_est",
           "real_est_fixed_opex_per_kW",
           "real_est_variable_opex_per_MWh",
           "real_fixed_opex_per_kW",
           "real_est_OpEx",
           "real_fuel_costs",
           "real_fuel_costs_per_MWh",
           "real_operating_costs",
           "real_operating_costs_per_kW",
           "real_operating_costs_per_MWh",
           "real_marginal_costs_per_MWh",
           "utility_id_eia"
)

#python_inputs_data <- python_inputs_data[keeps]

print("Writing python_inputs_data")
write_parquet(python_inputs_data,"python_inputs_data.parquet")




generation_fuel_table_hist <- historic_860_923_data %>%
  filter(!is.na(fuel_group_code)) %>%
  select(c(
    plant_id_eia,
    generator_id,
    report_year,
    report_month,
    prime_mover_code_hist,
    fuel_group_code,
    State,
    Balancing_Authority_Code,
    cap_tile,
    lat_tile,
    long_tile,
    month_hours,
    capacity_hist,
    unit_gen_code,
    net_generation_mwh_final,
    gross_gen_final,
    mmbtu_final,
    mmbtu_fg_final,
    fuel_frac,
    co2_fg,
    co2e_fg,
    gen_starts,
    fuel_starts,
    gen_starts_ave,
    fuel_starts_ave,
    final_fuel_cost_per_mmbtu
  )) %>% distinct() %>%
  mutate(
    gen_starts = ifelse(is.na(gen_starts),gen_starts_ave,gen_starts),
    fuel_starts = ifelse(is.na(fuel_starts),fuel_starts_ave,fuel_starts)
  ) %>%
  group_by(plant_id_eia,generator_id,report_year,report_month) %>%
  mutate(
    final_fuel_cost = final_fuel_cost_per_mmbtu * mmbtu_final * fuel_frac,
    final_fuel_cost_tot = sum(final_fuel_cost),
    co2_tot = sum(co2_fg),
    co2e_tot = sum(co2e_fg)
  ) %>% ungroup() %>%
  rename("co2" = "co2_fg", "co2e" = "co2e_fg") %>%
  left_join(FERC_Data %>% select(c(
    Plant,
    Prime,
    report_year,
    capex_per_kW,
    real_opex_per_kW,
    natural_gas,
    coal,
    petroleum,
    petroleum_coke,
    other_gas
  )) %>%
    pivot_longer(cols = c("natural_gas","coal","petroleum","petroleum_coke","other_gas"),
                 names_to = "fuel_group_code",
                 values_to = "FERC_fuel_frac",
                 values_drop_na = TRUE) %>%
    filter(FERC_fuel_frac>0), by = c(
      "prime_mover_code" = "Prime",
      "plant_id_eia" = "Plant",
      "report_year" = "report_year",
      "fuel_group_code" = "fuel_group_code"
    )) %>% distinct() %>%
  group_by(plant_id_eia,generator_id,report_year) %>%
  mutate(
    FERC_wt = ifelse(is.na(FERC_fuel_frac),0,FERC_fuel_frac * fuel_frac)/
      sum(ifelse(is.na(FERC_fuel_frac),0,FERC_fuel_frac * fuel_frac),na.rm=TRUE),
    capex_per_kW_sum = sum(FERC_wt * capex_per_kW, na.rm = TRUE),
    real_opex_per_kW_sum = sum(FERC_wt * real_opex_per_kW, na.rm = TRUE)
  ) %>%
  select(-c(FERC_fuel_frac,FERC_wt,capex_per_kW,real_opex_per_kW)) %>% distinct() %>%
  rename("capex_per_kW" = "capex_per_kW_sum","real_opex_per_kW" = "real_opex_per_kW_sum") %>%
  pivot_wider(names_from = fuel_group_code,
              names_glue = "{.value}_{fuel_group_code}",
              values_from = c(mmbtu_fg_final,fuel_frac,co2,co2e,final_fuel_cost),
              values_fn = list(mmbtu_fg_final = max,fuel_frac = max, co2 = max, co2e = max, final_fuel_cost = max),
              values_fill = 0
  )

# Now, map in the most recent actual FERC operating costs and capex if reported.

#pull EIA 923 generation and fuel data as well as fuel receipts data from 2008-2020

generation_fuel_table_annual_hist <- generation_fuel_table_hist %>%
  group_by(plant_id_eia, generator_id, report_year) %>%
  summarize(
    total_fuel_mmbtu = sum(ifelse(is.na(mmbtu_final),0,mmbtu_final)),
    total_fossil_mwh = sum(ifelse(is.na(net_generation_mwh_final),0,net_generation_mwh_final)),
    total_fossil_gross_mwh = sum(ifelse(is.na(gross_gen_final),0,gross_gen_final)),
    total_carbon = sum(ifelse(is.na(co2e_tot),0,co2e_tot)),
    total_CO2_emissions = sum(ifelse(is.na(co2_tot),0,co2_tot)),
    final_fuel_cost = sum(ifelse(is.na(final_fuel_cost_tot),0,final_fuel_cost_tot)),
    fuel_frac_natural_gas = ifelse(total_fuel_mmbtu>0,
                                   sum(ifelse(is.na(mmbtu_fg_final_natural_gas),0,
                                              mmbtu_fg_final_natural_gas))/total_fuel_mmbtu,0),
    fuel_frac_coal = ifelse(total_fuel_mmbtu>0,
                            sum(ifelse(is.na(mmbtu_fg_final_coal),0,
                                       mmbtu_fg_final_coal))/total_fuel_mmbtu,0),
    fuel_frac_petroleum = ifelse(total_fuel_mmbtu>0,
                                 sum(ifelse(is.na(mmbtu_fg_final_petroleum),0,
                                            mmbtu_fg_final_petroleum))/total_fuel_mmbtu,0),
    fuel_frac_petroleum_coke = ifelse(total_fuel_mmbtu>0,
                                      sum(ifelse(is.na(mmbtu_fg_final_petroleum_coke),0,
                                                 mmbtu_fg_final_petroleum_coke))/total_fuel_mmbtu,0),
    fuel_frac_other = ifelse(total_fuel_mmbtu>0,
                             sum(ifelse(is.na(mmbtu_fg_final_other),0,
                                        mmbtu_fg_final_other))/total_fuel_mmbtu,0),
    fuel_frac_other_gas = ifelse(total_fuel_mmbtu>0,
                                 sum(ifelse(is.na(mmbtu_fg_final_other_gas),0,
                                            mmbtu_fg_final_other_gas))/total_fuel_mmbtu,0),
    plant_id_eia = plant_id_eia,
    generator_id = generator_id,
    capacity_hist = capacity_hist,
    #    prime_mover_code = prime_mover_code_hist,
    #    fuel_code = fuel_group_code,
    #    utility_id_eia = utility_id_eia,
    report_year = report_year
    #    state = State
  ) %>% ungroup() %>% distinct()

#combine 860 and 923 data to create patio cost inputs table for integration of FERC data

python_inputs_data_hist <- unit_level_data %>%
  select(-c(report_month,month_hours)) %>% distinct() %>%
  select(-starts_with("rmi_")) %>%
  select(-starts_with("prime_mover_code_")) %>%
  select(-starts_with("fuel_code_")) %>%
  select(-starts_with("fuel_group_")) %>%
  select(-starts_with("energy_source_code_")) %>%
  left_join(generation_fuel_table_annual_hist, by = c(
    "plant_id_eia" = "plant_id_eia",
    "generator_id" = "generator_id",
    "report_year" = "report_year"
  ))

python_inputs_data_hist<-python_inputs_data_hist %>%
  group_by(plant_id_eia, generator_id) %>%
  mutate(
    fossil_CF = ifelse(capacity_hist>0,total_fossil_gross_mwh / (capacity_hist * ifelse(report_year %% 4 ==0,8784,8760)),0),
    median_CF = median(ifelse(fossil_CF<=1,fossil_CF,1),na.rm=TRUE),
    high_median_CF = ifelse(median_CF>0.6,1,0),
    mid_median_CF = ifelse(median_CF>0.4 & median_CF<=0.6,1,0),
    low_median_CF = ifelse(median_CF<=0.2,1,0),
    pollution_control_costs_per_kW = ifelse(is.na(pollution_control_costs_per_kW),0,pollution_control_costs_per_kW)
  ) %>% ungroup()

#merge python_inputs_data_hist and reg_variables
python_inputs_data_hist <- python_inputs_data_hist %>%
  left_join(reg_variables, by = c(
    "prime_mover_code" = "Prime"
  ))

#add state wage adjustment factors

python_inputs_data_hist <- python_inputs_data_hist %>%
  left_join(wage_factors, by = c(
    "State" = "State",
    "report_year" = "year"
  ))

# Calculate estimated capex per kW and annual maintenance capex costs, in nominal dollars - note that we
# break out the contribution to annual maintenance capex correlated with cumulative starts/stops to be
# dynamically calculated based on patio model starts and stops

python_inputs_data_hist <- python_inputs_data_hist %>%
  mutate(
    real_maint_capex_per_kW_no_cum_starts_est =
      -(age_obs_fixed +
          (gas_age_obs_fixed * fuel_frac_natural_gas) +
          (oil_age_obs_fixed * fuel_frac_petroleum) +
          (pulverized_coal_age_obs_fixed * pulverized_coal) +
          (supercritical_age_obs_fixed * supercritical) +
          (duct_burners_age_obs_fixed * duct_burners) +
          (bypass_hrsg_age_obs_fixed * bypass_hrsg)),
    real_maint_capex_per_kW_cum_starts_coeff = -cum_starts_age_obs_fixed,
    capex_per_kW_no_cum_starts_est =
      capacity +
      (gas_fixed * fuel_frac_natural_gas) +
      (oil_fixed * fuel_frac_petroleum) +
      (pollution_fixed * pollution_control_costs_per_kW) +
      (age_fixed * diff_age_and_avg_plant_prime_fuel) +
      (CHP_fixed * chp) +
      (pulverized_coal_fixed * pulverized_coal) +
      (other_comb_fixed * other_comb) +
      (ultrasupercritical_fixed * ultrasupercritical) +
      (gas_pollution_fixed * fuel_frac_natural_gas * pollution_control_costs_per_kW) +
      (gas_age_fixed * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel) +
      (oil_age_fixed * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel) +
      (gas_CHP_fixed * fuel_frac_natural_gas * chp) +
      (oil_CHP_fixed * fuel_frac_petroleum * chp) +
      #(duct_burners_fixed * duct_burners) +
      (bypass_hrsg_fixed * bypass_hrsg)
  )

# Calculate estimated real fixed costs opex as well as the start/stop costs. Note that we again break out
# the impact on real fixed opex of start/stops. Further, we break out the coefficient dependent on the age
# of observation for use only in historic cost estimation.

python_inputs_data_hist <- python_inputs_data_hist %>%
  mutate(real_fixed_opex_per_kW_no_starts_est = (
    (capacity_adj * wage_scale) +
      (median_CF_fixed_adj * median_CF * wage_scale ) +
      (high_median_CF_fixed_adj * high_median_CF * wage_scale ) +
      (mid_median_CF_fixed_adj * mid_median_CF * wage_scale ) +
      (low_median_CF_fixed_adj * low_median_CF * wage_scale ) +
      (gas_fixed_adj * fuel_frac_natural_gas * wage_scale) +
      (oil_fixed_adj * fuel_frac_petroleum * wage_scale) +
      (CHP_fixed_adj * chp * wage_scale) +
      (pollution_fixed_adj * pollution_control_costs_per_kW * wage_scale) +
      (pulverized_coal_fixed_adj * pulverized_coal * wage_scale) +
      (duct_burners_fixed_adj * duct_burners * wage_scale) +
      (supercritical_fixed_adj * supercritical * wage_scale) +
      (age_fixed_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (oil_age_fixed_adj * diff_age_and_avg_plant_prime_fuel * fuel_frac_petroleum * wage_scale) +
      (gas_pollution_fixed_adj * fuel_frac_natural_gas * pollution_control_costs_per_kW * wage_scale)),
    real_opex_per_kW_start = starts_adj +
      (gas_starts_adj * fuel_frac_natural_gas) +
      (oil_starts_adj * fuel_frac_petroleum),
    real_fixed_opex_per_kW_age_coeff = age_obs_fixed_adj * wage_scale
  )

# Calculate estimated real variable costs opex, again breaking out coefficients dependent on age of observation
# only for historical cost estimation

python_inputs_data_hist <- python_inputs_data_hist %>%
  mutate(real_variable_opex_per_MWh_est = (
    (gen_adj * wage_scale) +
      (median_CF_adj * median_CF * wage_scale ) +
      (high_median_CF_adj * high_median_CF * wage_scale ) +
      (mid_median_CF_adj * mid_median_CF * wage_scale ) +
      (low_median_CF_adj * low_median_CF * wage_scale ) +
      #(gas_variable_adj * fuel_frac_natural_gas * wage_scale) +
      (oil_variable_adj * fuel_frac_petroleum * wage_scale) +
      (pollution_variable_adj * pollution_control_costs_per_kW * wage_scale) +
      (age_variable_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (fluidized_bed_variable_adj * fluidized_bed * wage_scale) +
      (CHP_variable_adj * chp * wage_scale) +
      (supercritical_variable_adj * supercritical * wage_scale) +
      (gas_age_variable_adj * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (duct_burners_variable_adj * duct_burners * wage_scale) +
      (oil_age_variable_adj * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel * wage_scale)),
    real_variable_opex_per_MWh_age_coeff = (age_obs_variable_adj * wage_scale)
  )

python_inputs_data_hist <- python_inputs_data_hist %>%
  left_join(select(CPIU,c(Year,Inflation_Factor_2020)), by = c("report_year" = "Year")) %>%
  mutate(
    #real_fixed_opex_per_kW = ifelse(is.na(real_opex_per_kW),real_est_fixed_opex_per_kW,real_opex_per_kW) -
    #  real_est_variable_opex_per_MWh * ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020),
    #real_est_OpEx = capacity_adjusted * 1000 * ifelse(is.na(real_opex_per_kW),real_est_fixed_opex_per_kW,real_opex_per_kW) +
    #       real_est_variable_opex_per_MWh * (total_fossil_mwh -ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020)),
    real_fuel_costs = final_fuel_cost / Inflation_Factor_2020,
    real_fuel_costs_per_MWh = real_fuel_costs / total_fossil_mwh,
    #real_marginal_costs_per_MWh = real_fuel_costs_per_MWh + real_est_variable_opex_per_MWh,
    #real_operating_costs = real_est_OpEx + real_fuel_costs,
    #real_operating_costs_per_kW = real_operating_costs / (capacity_adjusted *1000),
    #real_operating_costs_per_MWh = real_operating_costs / total_fossil_mwh,
    emissions_intensity = total_carbon *1000000 / total_fossil_mwh
  )

#average_fossil_cost_BA <- python_inputs_data_hist %>%
#  select(c(Balancing_Authority_Code,total_fossil_mwh,report_year,real_operating_costs,total_carbon,real_operating_costs_per_MWh,emissions_intensity)) %>%
#  group_by(Balancing_Authority_Code,report_year) %>%
#  summarize(BA_OpEx = sum(real_operating_costs,na.rm = TRUE),
#            BA_gen = sum(total_fossil_mwh,na.rm = TRUE),
#            BA_emissions = sum(total_carbon,na.rm = TRUE)) %>%
#  ungroup %>%
#  unique %>%
#  mutate(average_BA_cost_per_mwh = BA_OpEx / BA_gen,
#         average_BA_emissions_intensity = BA_emissions / BA_gen)

#python_inputs_data_hist <- python_inputs_data_hist %>%
#  left_join(average_fossil_cost_BA,
#            by = c("Balancing_Authority_Code" = "Balancing_Authority_Code",
#                   "report_year" = "report_year"))

# Jettison intermediate data fields for subsequent analysis

keeps <- c("plant_id_eia",
           "prime_mover_code",
           "fuel_code",
           "report_year",
           "state",
           "Plant.Name",
           "Sector",
           "plant_prime_fuel_1",
           "Balancing_Authority_Code",
           "capacity_in_report_year",
           "capacity_adjusted",
           "capacity_adjustment",
           "total_fossil_mwh",
           "total_fuel_mmbtu",
           "total_carbon",
           "total_CO2_emissions",
           "final_fuel_cost",
           "fossil_CF",
           "median_CF",
           "capacity_in_2020",
           "capacity_in_2025",
           "capacity_in_2030",
           "capacity_in_2035",
           "average_BA_cost_per_mwh",
           "average_BA_emissions_intensity",
           "BA_OpEx",
           "BA_gen",
           "BA_emissions",
           "capex_per_kW",
           "real_opex_per_kW",
           "capex_per_kW_est",
           "maint_capex_per_kW_est",
           "real_est_fixed_opex_per_kW",
           "real_est_variable_opex_per_MWh",
           "real_fixed_opex_per_kW",
           "real_est_OpEx",
           "real_fuel_costs",
           "real_fuel_costs_per_MWh",
           "real_operating_costs",
           "real_operating_costs_per_kW",
           "real_operating_costs_per_MWh",
           "real_marginal_costs_per_MWh",
           "utility_id_eia"
)

#python_inputs_data_hist <- python_inputs_data_hist[keeps]

print("Writing python_inputs_data_hist")
write_parquet(python_inputs_data_hist,"python_inputs_data_hist.parquet")









## pull python data
python_data <- read.csv("patio_output_summary.csv",stringsAsFactors = TRUE)

python_data <- python_data %>% left_join(
  python_inputs_data %>% subset(select = -c(Plant.Name,Sector)),
  by = c(
    "plant_id_eia" = "plant_id_eia",
    "prime_mover_code" = "prime_mover_code",
    "fuel_code" = "fuel_code",
    "state" = "state",
    "utility_id_eia" = "utility_id_eia",
    "report_year" = "report_year"
  )
)


python_data <- python_data %>% left_join(
  ppf_capacity_table %>% subset(select = -c(Plant.Name,Sector)),
  by = c(
              "plant_id_eia" = "plant_id_eia",
              "prime_mover_code" = "prime_mover_code",
              "fuel_code" = "fuel_code",
              "state" = "state",
              "utility_id_eia" = "utility_id_eia",
              "report_year" = "report_year"
          )
  )

capacity_years = c(2020,2025,2030,2035)
for (capacity_year in capacity_years) {
  python_data <- python_data %>% left_join(
    ppf_capacity_table %>% filter(report_year == capacity_year) %>%
      subset(select = c(plant_id_eia,prime_mover_code,fuel_code,capacity_in_report_year)) %>%
      rename_with(~gsub("report_year",as.character(capacity_year),.x, fixed = TRUE)),
    by = c(
      "plant_id_eia" = "plant_id_eia",
      "prime_mover_code" = "prime_mover_code",
      "fuel_code" = "fuel_code"
    )
  )
}

python_data <- python_data %>%
  mutate(
    capacity_in_2025 = ifelse(capacity_in_2020>0,capacity_in_2025/capacity_in_2020,0),
    capacity_in_2030 = ifelse(capacity_in_2020>0,capacity_in_2030/capacity_in_2020,0),
    capacity_in_2035 = ifelse(capacity_in_2020>0,capacity_in_2035/capacity_in_2020,0)
  )

python_data<-python_data %>%
  group_by(plant_id_eia, prime_mover_code, fuel_code) %>%
  mutate(
    fossil_CF = ifelse(capacity>0,total_fossil_mwh / (capacity * ifelse(report_year %% 4 ==0,8784,8760)),0),
    median_CF = median(ifelse(fossil_CF<=1,fossil_CF,NA),na.rm=TRUE),
    high_median_CF = ifelse(median_CF>0.6,1,0),
    mid_median_CF = ifelse(median_CF>0.4 & median_CF<=0.6,1,0),
    low_median_CF = ifelse(median_CF<=0.2,1,0))

python_data <- python_data %>%
  left_join(python_data %>% select(c(plant_id_eia,prime_mover_code,fuel_code,report_year,pct_of_re,total_fossil_mwh,fossil_CF)) %>%
              filter(report_year==2020), by = c(
    "plant_id_eia" = "plant_id_eia",
    "prime_mover_code" = "prime_mover_code",
    "fuel_code" = "fuel_code",
    "pct_of_re" = "pct_of_re"
  ), suffix = c("","_2020")) %>% select(-c(report_year_2020))
python_data <- python_data %>%
  filter(total_fossil_mwh > 0)

python_data$clean_fraction<-python_data$pct_of_re/100
python_data$pollution_control_costs_per_kW<-ifelse(is.na(python_data$pollution_control_costs_per_kW),0,python_data$pollution_control_costs_per_kW)
python_data$remaining_fossil_generation<-python_data$total_fossil_mwh - python_data$avoided_fossil_mwh
python_data$remaining_fossil_CF<-python_data$remaining_fossil_generation/(python_data$capacity * ifelse(python_data$report_year %% 4 ==0,8784,8760))

#python_data <- python_data %>% left_join(select(CPIU,c(Year,Inflation_Factor_2020)), by = c("report_year" = "Year"))

#merge python_data and reg_variables
python_data <- merge(python_data, reg_variables,by.x="prime_mover_code", by.y="Prime")

#merge in FERC data for the current year to use current actuals for FERC reporting plant-primes

FERC_Data_current<-FERC_Data %>% subset(report_year == 2020)

python_data <- merge(python_data, FERC_Data_current[,c("Plant", "Prime",
                                                       "capex_per_kW", "real_opex_per_kW","Fuel 1")],
                     by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                     by.y=c("Plant", "Prime","Fuel 1"), all.x=TRUE)
python_data <- python_data %>%
  rename(c(capex_per_kW1 = capex_per_kW,
           real_opex_per_kW1 = real_opex_per_kW))

python_data <- merge(python_data, FERC_Data_current[,c("Plant", "Prime",
                                                       "capex_per_kW", "real_opex_per_kW","Fuel 2")],
                     by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                     by.y=c("Plant", "Prime","Fuel 2"), all.x=TRUE)
python_data <- python_data %>%
  rename(c(capex_per_kW2 = capex_per_kW,
           real_opex_per_kW2 = real_opex_per_kW))

python_data <- merge(python_data, FERC_Data_current[,c("Plant", "Prime",
                                                       "capex_per_kW", "real_opex_per_kW","Fuel 3")],
                     by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                     by.y=c("Plant", "Prime","Fuel 3"), all.x=TRUE)
python_data <- python_data %>%
  rename(c(capex_per_kW3 = capex_per_kW,
           real_opex_per_kW3 = real_opex_per_kW))

python_data <- python_data %>%
  mutate(capex_per_kW = coalesce(capex_per_kW1,capex_per_kW2,capex_per_kW3),
         real_opex_per_kW = coalesce(real_opex_per_kW1,real_opex_per_kW2,real_opex_per_kW3)) %>%
  select(-c(capex_per_kW1,capex_per_kW2,capex_per_kW3,
            real_opex_per_kW1,real_opex_per_kW2,real_opex_per_kW3))

#add state wage adjustment for current year

wage_factors_recent<-wage_factors %>% subset(year==2020)
python_data <- merge(python_data, wage_factors_recent[,c("State","wage_scale")],
                     by.x=c("state"),
                     by.y=c("State"), all.x=TRUE)

#sum(is.na(python_data$wage_scale))
#sum(is.na(FERC_Data$wage_scale))
#test_nas <- python_data[is.na(python_data$wage_scale),]

#calculate estimated capex per kW and annual maintenance capex costs, in nominal dollars

python_data <- python_data %>%
  mutate(capex_per_kW_est =
           (capacity.y +
              (gas_fixed * fuel_frac_natural_gas) +
              (oil_fixed * fuel_frac_petroleum) +
              (other_fixed * fuel_frac_other_gas) +
              (pollution_fixed * pollution_control_costs_per_kW) +
              (age_fixed * diff_age_and_avg_plant_prime_fuel) +
              (CHP_fixed * chp) +
              (fluidized_bed_fixed * fluidized_bed) +
              (other_comb_fixed * other_comb) +
              (ultrasupercritical_fixed * ultrasupercritical) +
              (gas_pollution_fixed * fuel_frac_natural_gas * pollution_control_costs_per_kW) +
              (gas_age_fixed * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel) +
              (oil_age_fixed * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel) +
              (gas_CHP_fixed * fuel_frac_natural_gas * chp) +
              (oil_CHP_fixed * fuel_frac_petroleum * chp) +
              (duct_burners_fixed * duct_burners) +
              (bypass_hrsg_fixed * bypass_hrsg)),
         maint_capex_per_kW_est = age_fixed + (gas_age_fixed * fuel_frac_natural_gas) + (oil_age_fixed * fuel_frac_petroleum)
         )

#python_data <- python_data %>%
#  mutate(original_cost = capacity.x * 1000 * ifelse(is.na(capex_per_kW),capex_per_kW_est,capex_per_kW))

#calculate estimated real fixed costs opex

python_data <- python_data %>%
  mutate(cur_est_fixed_opex_per_kW = (
    (capacity_adj * wage_scale) +
      (median_CF_fixed_adj * median_CF * wage_scale ) +
      (gas_fixed_adj * fuel_frac_natural_gas * wage_scale) +
      (oil_fixed_adj * fuel_frac_petroleum * wage_scale) +
      (pollution_fixed_adj * pollution_control_costs_per_kW * wage_scale) +
      (age_fixed_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (gas_pollution_fixed_adj * fuel_frac_natural_gas * pollution_control_costs_per_kW * wage_scale)))

#calculate estimated real variable costs opex

python_data <- python_data %>%
  mutate(cur_est_variable_opex_per_MWh = (
    (gen_adj * wage_scale) +
      (median_CF_adj * median_CF * wage_scale ) +
      (mid_median_CF_adj * mid_median_CF * wage_scale ) +
      #(gas_variable_adj * fuel_frac_natural_gas * wage_scale) +
      #(oil_variable_adj * fuel_frac_petroleum * wage_scale) +
      (pollution_variable_adj * pollution_control_costs_per_kW * wage_scale) +
      (age_variable_adj * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      #(fluidized_bed_variable_adj * fluidized_bed * wage_scale) +
      (supercritical_variable_adj * supercritical * wage_scale) +
      (gas_age_variable_adj * fuel_frac_natural_gas * diff_age_and_avg_plant_prime_fuel * wage_scale) +
      (CHP_variable_adj * chp * wage_scale) +
      (duct_burners_variable_adj * duct_burners * wage_scale) +
      (oil_age_variable_adj * fuel_frac_petroleum * diff_age_and_avg_plant_prime_fuel * wage_scale)))


python_data <- python_data %>%
  mutate(current_est_OpEx = capacity.x * 1000 * ifelse(is.na(real_opex_per_kW),cur_est_fixed_opex_per_kW,real_opex_per_kW) +
                               cur_est_variable_opex_per_MWh * (total_fossil_mwh -ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020)),
         remaining_est_OpEx = capacity.x * 1000 * ifelse(is.na(real_opex_per_kW),cur_est_fixed_opex_per_kW,real_opex_per_kW) +
                               cur_est_variable_opex_per_MWh * (remaining_fossil_generation - ifelse(is.na(real_opex_per_kW),0,total_fossil_mwh_2020)),
         current_est_fuel_cost = (cost_per_mmbtu_total/Inflation_Factor_2020) * total_fuel_mmbtu,
         remaining_est_fuel_cost = current_est_fuel_cost - (cost_per_mmbtu_avoided/Inflation_Factor_2020) * avoided_fuel_mmbtu,
         current_operating_costs = current_est_OpEx + current_est_fuel_cost,
         remaining_operating_costs = remaining_est_OpEx + remaining_est_fuel_cost,
         current_operating_costs_per_kW = current_operating_costs / (capacity.x *1000),
         remaining_operating_costs_per_kW = remaining_operating_costs / (capacity.x * 1000),
         current_fossil_operating_cost_per_MWh = current_operating_costs / total_fossil_mwh,
         remaining_fossil_operating_cost_per_MWh = remaining_operating_costs / total_fossil_mwh,
         operating_cost_savings_per_MWh = current_fossil_operating_cost_per_MWh - remaining_fossil_operating_cost_per_MWh,
         renewable_to_fossil_gen_ratio = (solar_mwh + onshore_wind_mwh + offshore_wind_mwh)/total_fossil_mwh,
         emissions_intensity = total_carbon *1000000 / total_fossil_mwh
         #breakeven_clean_costs_per_MWh = operating_cost_savings_per_MWh/renewable_to_fossil_gen_ratio,
         #onshore_wind_CF = ifelse(onshore_wind>0,onshore_wind_mwh / (onshore_wind * ifelse(report_year %% 4 ==0,8784,8760)),0),
         #offshore_wind_CF = ifelse(offshore_wind>0,offshore_wind_mwh / (offshore_wind * ifelse(report_year %% 4 ==0,8784,8760)),0),
         #solar_CF = ifelse(solar>0,solar_mwh / (solar * ifelse(report_year %% 4 ==0,8784,8760)),0),
        )

average_fossil_cost_BA <- python_data %>%
  filter(clean_fraction == 1) %>%
  select(c(Balancing_Authority_Code,total_fossil_mwh,report_year,current_operating_costs,total_carbon,current_fossil_operating_cost_per_MWh,emissions_intensity)) %>%
  group_by(Balancing_Authority_Code,report_year) %>%
  summarize(BA_OpEx = sum(current_operating_costs),
            BA_gen = sum(total_fossil_mwh),
            BA_emissions = sum(total_carbon)) %>%
  ungroup %>%
  unique %>%
  mutate(average_BA_cost_per_mwh = BA_OpEx / BA_gen,
         average_BA_emissions_intensity = BA_emissions / BA_gen)

python_data <- python_data %>%
  left_join(average_fossil_cost_BA,
            by = c("Balancing_Authority_Code" = "Balancing_Authority_Code",
                   "report_year" = "report_year"))

#python_data <- python_data %>%
#  filter(!is.na(breakeven_clean_costs_per_MWh))

## match utility IDs and ownership

MUL_Cols <- as.character(read_excel("Fossil Master Unit List.xlsx",
                                    skip = 2,
                                    n_max = 1,
                                    col_names = FALSE))

master_unit_list <- read_excel("Fossil Master Unit List.xlsx",
                               skip = 4,
                               col_names = MUL_Cols,
                               guess_max = 20000)

master_unit_list <- master_unit_list[,c("Plant_Code",
                                        "Prime",
                                        "Primary_FUEL_GROUP",
                                        "Utility_ID",
                                        #"Balancing_Authority_Code",
                                        #"Percent_Owned",
                                        "plant_prime_fuel",
                                        "plant_prime_fuel_utility",
                                        "entity_type",
                                        "owned_capacity")] %>%
  group_by(plant_prime_fuel_utility) %>%
  summarise(owned_capacity = sum(owned_capacity),
            Plant_Code,
            Prime,
            Primary_FUEL_GROUP,
            Utility_ID,
            #Balancing_Authority_Code,
            plant_prime_fuel,
            entity_type,
            .groups="drop") %>%
  distinct(.keep_all = TRUE)
master_unit_list <- as.data.frame(unclass(master_unit_list),stringsAsFactors = TRUE)

python_data <- merge(python_data, master_unit_list,
                     by.x=c("plant_id_eia",
                            "prime_mover_code",
                            "fuel_code"),
                     by.y=c("Plant_Code",
                            "Prime",
                            "Primary_FUEL_GROUP"),
                     all.x=TRUE)

python_data <- python_data %>% subset(owned_capacity > 0)

python_data <- python_data %>%
  arrange(plant_id_eia)

# Jettison intermediate data fields for subsequent analysis

keeps <- c("plant_id_eia",
           "prime_mover_code",
           "fuel_code",
           "report_year",
           "clean_fraction",
           "state",
           "Balancing_Authority_Code",
           "capacity.x",
           "capacity_in_report_year",
           "capacity_in_2020",
           "capacity_in_2025",
           "capacity_in_2030",
           "capacity_in_2035",
           "onshore_wind",
           "solar",
           "offshore_wind",
           "avoided_fossil_mwh",
           "avoided_fuel_mmbtu",
           "avoided_carbon",
           "total_fossil_mwh",
           "total_fuel_mmbtu",
           "total_carbon",
           "emissions..CO2e.",
           "fossil_CF",
           "remaining_fossil_generation",
           "remaining_fossil_CF",
           "onshore_wind_mwh",
           "solar_mwh",
           "offshore_wind_mwh",
           "excess_onshore_wind_mwh",
           "excess_solar_mwh",
           "excess_offshore_wind_mwh",
           "distance",
           "cost_per_mwh_avoided",
           "cost_per_mmbtu_avoided",
           "cost_per_mwh_total",
           "cost_per_mmbtu_total",
           "average_BA_cost_per_mwh",
           "average_BA_emissions_intensity",
           "BA_OpEx",
           "BA_gen",
           "BA_emissions",
           "capex_per_kW",
           "capex_per_kW_est",
           "maint_capex_per_kW_est",
           "current_operating_costs",
           "remaining_operating_costs",
           "owned_capacity",
           "Utility_ID",
           "utility_id_eia",
           "entity_type",
           "Percent_Owned"
           )

python_data <- python_data %>%
  mutate(
    Percent_Owned = owned_capacity / capacity.x
    #owned_original_cost = Percent_Owned * original_cost
    )

#complete_python_data <- python_data
#print("Writing complete_python_data")
#write_parquet(complete_python_data,"complete_python_data.parquet")

python_data <- python_data[keeps] %>%
  mutate(technology = as.factor(ifelse(prime_mover_code == "ST", "steam", "other_fossil"))) %>%
  rename(capacity = capacity.x,
         operating_utility_id = utility_id_eia)

## Merge in state-level NREL and tax data

state_mapping <- read_excel("State_Data.xlsx",
                            sheet = "States")
state_mapping <- as.data.frame(unclass(state_mapping), stringsAsFactors = TRUE, optional = TRUE)

python_data <- python_data %>% left_join(state_mapping[,c("State",
                                                          "NREL_Scenario",
                                                          "NREL_Wind_Class",
                                                          "NREL_Solar_Class",
                                                          "NREL_Offshore_Wind_Class",
                                                          "State_Tax_Rate")],
                                         by = c("state" = "State"))

print("Writing python_data")
write_parquet(python_data,"python_data.parquet")



## Import and transform Hub data to integrate utility financial data for calculation of revenue requirements implications
## of new asset deployment as well as existing asset securitization

files <- c("net_plant_balance", "debt_equity_returns", "revenue_by_tech", "operations_emissions_by_tech")
for (file in files) {
  filename = paste(file,'.csv',sep="")
  if (file.exists(filename)) print(paste(filename,' already downloaded')) else download.file(paste('https://utilitytransitionhub.rmi.org/static/data_download/',filename,sep=""),filename)
  assign(paste("hub",file,sep="_"),read.csv(filename, stringsAsFactors = TRUE))
  #  assign(paste("hub",file,"2020",sep="_"),read.csv(filename) %>% filter(year == 2020))
}
aggregate_hub_revenue_by_tech <- aggregate(. ~ respondent_id + year + technology + component,
                                           data = (hub_revenue_by_tech %>%
                                                     select(-c(parent_name,utility_name,detail))), sum)

hub_data <- left_join(hub_net_plant_balance,
                      select(aggregate_hub_revenue_by_tech %>% filter(component == "returns"),
                             c(respondent_id,year,technology,revenue_total)),
                      by=c("respondent_id" = "respondent_id",
                           "year" = "year",
                           "FERC_class" = "technology")) %>%
  rename(returns = revenue_total)

hub_data <- left_join(hub_data,
                      select(aggregate_hub_revenue_by_tech %>% filter(component == "depreciation_expense"),
                             c(respondent_id,year,technology,revenue_total)),
                      by=c("respondent_id" = "respondent_id",
                           "year" = "year",
                           "FERC_class" = "technology")) %>%
  rename(depreciation_expense = revenue_total)

hub_data <- left_join(hub_data,
                      select(aggregate_hub_revenue_by_tech %>% filter(component == "maintenance_expenses"),
                             c(respondent_id,year,technology,revenue_total)),
                      by=c("respondent_id" = "respondent_id",
                           "year" = "year",
                           "FERC_class" = "technology")) %>%
  rename(maintenance_expenses = revenue_total)

hub_data <- left_join(hub_data,
                      select(aggregate_hub_revenue_by_tech %>% filter(component == "non_fuel_operation_expenses"),
                             c(respondent_id,year,technology,revenue_total)),
                      by=c("respondent_id" = "respondent_id",
                           "year" = "year",
                           "FERC_class" = "technology")) %>%
  rename(non_fuel_operation_expenses = revenue_total)

hub_data <- left_join(hub_data,
                      select(aggregate_hub_revenue_by_tech %>% filter(component == "depreciation_expense_for_asset_retirement_costs"),
                             c(respondent_id,year,technology,revenue_total)),
                      by=c("respondent_id" = "respondent_id",
                           "year" = "year",
                           "FERC_class" = "technology")) %>%
  rename(depreciation_expense_ARC = revenue_total)

aggregate_hub_data <- hub_data %>% subset(original_cost > 0 &
                                            accum_depr > 0 &
                                            depreciation_expense > 0 &
                                            maintenance_expenses > 0 &
                                            non_fuel_operation_expenses > 0,
                                          select = -c(parent_name,
                                                      utility_name,
                                                      respondent_id,
                                                      ARC,
                                                      ARC_accum_depr,
                                                      net_ARC,
                                                      returns,
                                                      depreciation_expense_ARC))
aggregate_hub_data <- aggregate( . ~ FERC_class + year, data = aggregate_hub_data,sum)
aggregate_hub_data <- aggregate_hub_data %>%
  mutate(frac_depr = accum_depr / original_cost,
         depreciation_rate = depreciation_expense / original_cost,
         fixed_OM_frac = (non_fuel_operation_expenses+maintenance_expenses) / original_cost,
         rem_life = (1-frac_depr)/depreciation_rate)

#hub_operations_emissions_by_tech_by_utility_tech <- hub_operations_emissions_by_tech %>%
#  group_by(respondent_id,year,technology_RMI) %>%
#  summarise(emissions_co2 = sum(emissions_co2),
#            generation = sum(generation)) %>%
#  mutate(emissions_intensity = emissions_co2 / generation) %>%
#  rename(technology = technology_RMI)
#hub_operations_emissions_by_tech_by_utility_tech$technology <- gsub(" ", "_", hub_operations_emissions_by_tech_by_utility_tech$technology)
#hub_operations_emissions_by_tech_by_utility_tech$technology <- tolower(hub_operations_emissions_by_tech_by_utility_tech$technology)

## Utility Inputs

# can't use hub_utilities_download because need coop/muni manual data, but did update with newest data from download
#hub_utilities_download <- getURL("https://utilitytransitionhub.rmi.org/static/data_download/utility_information.csv")
#hub_utilities <- read.csv(text = hub_utilities_download)
#hub_utilities <- hub_utilities %>%
#  select(parent_name, utility_name, respondent_id, utility_id_eia, entity_type_eia, utility_type_rmi)

## Import utility information sheet and integrate state-by-state tax
eia_ownership_ids <- python_data %>%
  filter(owned_capacity > 0) %>%
  distinct(Utility_ID, technology, state, entity_type)

utilities_inputs <- read_excel("utility_information.xlsx",sheet="utility_information")
utilities_inputs <- as.data.frame(unclass(utilities_inputs), stringsAsFactors = TRUE, optional = TRUE)
utilities_inputs <- left_join(eia_ownership_ids, utilities_inputs,
                              by=c("Utility_ID" = "utility_id_eia"))
utilities_inputs <- utilities_inputs %>%
  mutate(state = coalesce(state.x,state.y)) %>%
  select(-c(state.x,state.y)) %>%
  arrange(respondent_id)

coop_data <- read_excel("Co-Op Analysis Summary Sheet.xlsx",skip=1) %>%
  clean_names() %>%
  rename(Utility_ID = utility_id,
         roe_2 = roe,
         equity_ratio_2 = equity_ratio,
         ror_2 = overall_ror) %>%
  drop_na(Utility_ID)

aggregate_coop_data <- coop_data %>% subset(!is.na(roe_2) &
                                              !is.na(equity_ratio_2) &
                                              !is.na(ror_2),
                                            select = c(Utility_ID,
                                                       utility_name,
                                                       equity_thousands,
                                                       debt_thousands,
                                                       roe_2,
                                                       equity_ratio_2,
                                                       ror_2)) %>%
  mutate(utility_coop_financial = equity_thousands + debt_thousands)
total_coop_financial = sum(aggregate_coop_data$utility_coop_financial)
aggregate_coop_data<- aggregate_coop_data %>%
  mutate(proportion = utility_coop_financial / total_coop_financial,
         proportion_roe_2 = roe_2 * proportion,
         proportion_equity_ratio_2 = equity_ratio_2 * proportion,
         proportion_ror_2 = ror_2 * proportion)

average_coop_roe = sum(aggregate_coop_data$proportion_roe_2)
average_coop_equity_ratio = sum(aggregate_coop_data$proportion_equity_ratio_2)
average_coop_ror = sum(aggregate_coop_data$proportion_ror_2)

utilities_inputs <- left_join(utilities_inputs, select(coop_data, c("Utility_ID",
                                                                    "roe_2",
                                                                    "equity_ratio_2",
                                                                    "ror_2")),
                              by="Utility_ID")

## Transmission costs

## Import cleaned transmission line cost and characteristics data from FERC Form 1, matched to estimate transmission line
## parameters via a python module. Then estimate Surge Impedence Loading (SIL) based on line parameters. Finally, based on
## brutally averaging a St. Clair curve, approximate line capacity as 1.5 x SIL

CPIU <- CPIU[c("Year","Inflation_Factor_2020")]

voltage_spacing<-read_excel("Voltage_Spacing_3phasee.xlsx")
transmission_capex<-read_excel("../pudl_outputs/f1_transmission_cleaned.xlsx",guess_max = 20000) %>%
  filter(row_literal != "TOTAL"
         & cost_land + cost_poles + cost_cndctr > 1000000
         & crct_present > 0
         & line_length * crct_present >= 10
         & voltage > 0) %>%
  left_join(CPIU,by=c("year"="Year")) %>%
  left_join(voltage_spacing,by=c("voltage"="voltage")) %>%
  mutate(voltage = ifelse(voltage > 1000,voltage/1000,voltage),
         cable_radius_ft = diameter_inch_comp_cable_OD/24,
         resistance_per_mile = resistance_ohms_kft_AC_75C * 5.28,
         bundles = ifelse(pmax(1,pmin(cnd_size_1,cnd_size_2,cnd_size_3,cnd_size_4,cnd_size_5,na.rm=TRUE),na.rm=TRUE) >=8,
                          1,pmax(1,pmin(cnd_size_1,cnd_size_2,cnd_size_3,cnd_size_4,cnd_size_5,na.rm=TRUE),na.rm=TRUE)),
         GMR_ft = cable_radius_ft * 0.7788, # Geometric mean radius of the composite cable accounting for inductance within cable
         GMR_bundle = (bundles * GMR_ft * ifelse(bundles>1,(1.5/(2*sinpi(1/bundles)))^(bundles-1),1)),
         r_bundle = (bundles * cable_radius_ft * ifelse(bundles>1,(1.5/(2*sinpi(1/bundles)))^(bundles-1),1)),
         X_L = 2.022 * 10^(-3) * 60 * log(spacing_ft/GMR_bundle),
         X_C = 1.779 * 10^6 * (1/60) * log(spacing_ft/cable_radius_ft),
         Z_0 = (X_L * X_C)^(1/2),
         SIL = crct_present * voltage^2 / Z_0,
         line_capacity_thermal_limit = crct_present * 3^(1/2) * ampacity * voltage / 1000,
         line_capacity = 1.5 * SIL
  )

## OLD cleaning of transmission line data in R
#transmission_capex<-read.csv("../pudl_outputs/f1_added_transmission_per_year_unaggregated.csv") %>%
#  filter(row_literal != "TOTAL" & cost_land + cost_poles + cost_cndctr > 0 & crct_present > 0 )
#transmission_capex$cnd_size<-gsub("[()]2[()]|2x|2[-]|T2[-]|(30w[-])|(6W[-])|(12W[-])|TP|T2|2[*]|4/0|000|([[:alpha:]][[:alpha:]]\\d\\d)|[[:alpha:]]|(\\d\\s-\\s)|(\\d\\s-)|(\\d-)|(12-)|26[/]7|24[/]7|3[/]0|[()]|1113,\\s|[/]477|,|\\s","",transmission_capex$cndctr_size)
#transmission_capex$cnd_size<-as.numeric(transmission_capex$cnd_size)
#transmission_capex$cnd_size<-ifelse(transmission_capex$cnd_size >10000,transmission_capex$cnd_size/1000000,transmission_capex$cnd_size)
#transmission_capex$cnd_size<-ifelse(transmission_capex$cnd_size <10,transmission_capex$cnd_size*100,transmission_capex$cnd_size)
#transmission_capex<-transmission_capex %>%
#  filter(line_length * crct_present >= 10) %>%
#  left_join(CPIU,by=c("year"="Year"))

## Estimate transmission line costs per mile x kW line capacity based on FERC costs and line capacity estimates

transmission_capex<-transmission_capex[,c("respondent_id",
                                          "year",
                                          "line_length",
                                          "crct_present",
                                          "voltage",
                                          "line_capacity",
                                          "cost_land",
                                          #                                          "cnd_size",
                                          "cost_poles",
                                          "cost_cndctr",
                                          "asset_retire_cost",
                                          "Inflation_Factor_2020",
                                          "cost_total")] %>%
  mutate(#line_crct = line_length * crct_present,
    #line_crct_size = line_crct * cnd_size,
    cost_build = (cost_land + cost_poles + cost_cndctr)/Inflation_Factor_2020)

aggregate_trans_data <- transmission_capex %>%
  subset(
    line_length > 0 &
      line_capacity > 0 &
      cost_build >0,
    select = c(respondent_id, line_length, line_capacity, cost_build))
aggregate_trans_data <- aggregate( . ~ respondent_id, data = aggregate_trans_data,sum) %>%
  mutate(Transmission_CAPEX = cost_build/ (line_length * line_capacity *1000))

transmission_fixed_OM_frac = aggregate_hub_data %>%
  subset(FERC_class == "transmission" & year == 2020,fixed_OM_frac) %>% as.numeric
average_transmission_CAPEX = sum(aggregate_trans_data$cost_build, na.rm = TRUE) / (sum(aggregate_trans_data$line_capacity, na.rm = TRUE) * sum(aggregate_trans_data$line_length, na.rm = TRUE))

# Merge historical transmission O&M and depreciation data
utilities_inputs <- left_join(utilities_inputs,
                              select(hub_data %>% filter(year==2020, FERC_class == "transmission")
                                     , c(respondent_id,
                                         original_cost,
                                         non_fuel_operation_expenses,
                                         depreciation_expense,
                                         maintenance_expenses)),
                              by=c("respondent_id" = "respondent_id")) %>%
  rename(transmission_original_cost = original_cost,
         transmission_depreciation_expense = depreciation_expense)

# Merge net plant balances, returns, depreciation expenses
utilities_inputs <- left_join(utilities_inputs,
                              select(hub_data %>% filter(year==2020)
                                     , c(respondent_id,
                                         FERC_class,
                                         original_cost,
                                         accum_depr,
                                         net_plant_balance,
                                         returns,
                                         depreciation_expense)),
                              by=c("respondent_id" = "respondent_id",
                                   "technology" = "FERC_class"))

# Calculate average depreciation rates
steam_average_depreciation_rate = aggregate_hub_data %>%
  subset(FERC_class == "steam" & year == 2020,depreciation_rate) %>% as.numeric
other_fossil_average_depreciation_rate = aggregate_hub_data %>%
  subset(FERC_class == "other_fossil" & year == 2020,depreciation_rate) %>% as.numeric
transmission_average_depreciation_rate = aggregate_hub_data %>%
  subset(FERC_class == "transmission" & year == 2020,depreciation_rate) %>% as.numeric


# Insert depreciation rates
utilities_inputs <- utilities_inputs %>%
  mutate(depreciation_rate = case_when(
    depreciation_expense / original_cost == 0 & technology == "steam" ~ steam_average_depreciation_rate,
    is.na(depreciation_expense / original_cost) & technology == "steam" ~ steam_average_depreciation_rate,
    depreciation_expense / original_cost == 0 & technology == "other_fossil" ~ other_fossil_average_depreciation_rate,
    is.na(depreciation_expense / original_cost) & technology == "other_fossil" ~ other_fossil_average_depreciation_rate,
    depreciation_expense / original_cost != 0 ~ abs(depreciation_expense / original_cost)))

# Calculate average fraction of original costs that have been depreciated
steam_average_frac_depr = aggregate_hub_data %>%
  subset(FERC_class == "steam" & year == 2020,frac_depr) %>% as.numeric
other_fossil_average_frac_depr = aggregate_hub_data %>%
  subset(FERC_class == "other_fossil" & year == 2020,frac_depr) %>% as.numeric
transmission_frac_depr = aggregate_hub_data %>%
  subset(FERC_class == "transmission" & year == 2020,frac_depr) %>% as.numeric

# Insert average fraction of original costs that have been depreciated
utilities_inputs <- utilities_inputs %>%
  mutate(frac_depr = case_when(
    accum_depr / original_cost == 0 & technology == "steam" ~ steam_average_frac_depr,
    is.na(accum_depr / original_cost) & technology == "steam" ~ steam_average_frac_depr,
    accum_depr / original_cost == 0 & technology == "other_fossil" ~ other_fossil_average_frac_depr,
    is.na(accum_depr / original_cost) & technology == "other_fossil" ~ other_fossil_average_frac_depr,
    accum_depr / original_cost != 0 ~ abs(accum_depr / original_cost)))

# Find net plant balance by utility
hub_net_plant_balance_2020_by_utility <- hub_net_plant_balance %>%
  filter(year == 2020) %>%
  group_by(respondent_id) %>%
  summarise(net_plant_balance = sum(net_plant_balance))

#Read in FERC Rate Case Data
ferc_SP_utility_map <- read_excel("ferc_SP_utility_map.xlsx",
                                  sheet = "ferc_SP_utility_map") %>%
  arrange(`FERC Utility ID`) %>%
  drop_na("FERC Utility ID")

# Calculate ror, roe, equity ratio
ferc_SP_utility_map <- left_join(ferc_SP_utility_map,hub_net_plant_balance_2020_by_utility,
                                 by=c("FERC Utility ID" = "respondent_id"))
ferc_SP_utility_map <- ferc_SP_utility_map %>%
  mutate(ror_plant_balance = `Most Recent ROR` * net_plant_balance,
         ror_plant_balance = ifelse(is.na(ror_plant_balance), 0, ror_plant_balance),
         roe_plant_balance = `Most Recent ROE` * net_plant_balance,
         roe_plant_balance = ifelse(is.na(roe_plant_balance), 0, roe_plant_balance),
         equity_ratio_plant_balance = `Most Recent Equity Ratio` * net_plant_balance,
         equity_ratio_plant_balance = ifelse(is.na(equity_ratio_plant_balance), 0, equity_ratio_plant_balance),
         net_plant_balance = ifelse(is.na(net_plant_balance), 0, net_plant_balance))

# Calculate average ror, roe, equity ratio
average_ror <- sum(ferc_SP_utility_map$ror_plant_balance) / sum(ferc_SP_utility_map$net_plant_balance)
average_roe <- sum(ferc_SP_utility_map$roe_plant_balance) / sum(ferc_SP_utility_map$net_plant_balance)
average_equity_ratio <- sum(ferc_SP_utility_map$equity_ratio_plant_balance) / sum(ferc_SP_utility_map$net_plant_balance)

# Merge in ror, roe, equity ratio
utilities_inputs <- left_join(utilities_inputs,
                              select(ferc_SP_utility_map,c("FERC Utility ID",
                                                           "Most Recent ROR",
                                                           "Most Recent ROE",
                                                           "Most Recent Equity Ratio")),
                              by=c("respondent_id" = "FERC Utility ID" ))
utilities_inputs <- utilities_inputs %>%
  rename(ror_1 = `Most Recent ROR`,
         roe_1 = `Most Recent ROE`,
         equity_ratio_1 = `Most Recent Equity Ratio`) %>%
  mutate(ror_1 = case_when(!is.na(ror_1) ~ ror_1,
                           is.na(ror_1) & !is.na(ror_2) ~ ror_2,
                           is.na(ror_1) & is.na(ror_2) & entity_type == "C" ~ average_coop_ror,
                           is.na(ror_1) & is.na(ror_2) & entity_type != "C" ~ average_ror),
         roe_1 = case_when(!is.na(roe_1) ~ roe_1,
                           is.na(roe_1) & !is.na(roe_2) ~ roe_2,
                           is.na(roe_1) & is.na(roe_2) & entity_type == "C" ~ average_coop_roe,
                           is.na(roe_1) & is.na(roe_2) & entity_type != "C" ~ average_roe),
         equity_ratio_1 = case_when(!is.na(equity_ratio_1) ~ equity_ratio_1,
                                    is.na(equity_ratio_1) & !is.na(equity_ratio_2) ~ equity_ratio_2,
                                    is.na(equity_ratio_1) & is.na(equity_ratio_2) & entity_type == "C" ~ average_coop_equity_ratio,
                                    is.na(equity_ratio_1) & is.na(equity_ratio_2) & entity_type != "C" ~ average_equity_ratio))

utilities_inputs <- utilities_inputs %>%
  mutate(roe = coalesce(roe_1, roe_2),
         equity_ratio = coalesce(equity_ratio_1, equity_ratio_2),
         ror = coalesce(ror_1, ror_2)) %>%
  select(-c(roe_1, roe_2, equity_ratio_1, equity_ratio_2, ror_1, ror_2))

#Merge in and calculate transmission capital cost data from FERC

utilities_inputs <- left_join(utilities_inputs,
                              select(aggregate_trans_data, c(respondent_id, Transmission_CAPEX)),
                              by = ("respondent_id" = "respondent_id")) %>%
  mutate(
    Transmission_CAPEX = ifelse(is.na(Transmission_CAPEX),average_transmission_CAPEX,Transmission_CAPEX),
    transmission_fixed_OM_frac = case_when(
      (maintenance_expenses + non_fuel_operation_expenses) / transmission_original_cost > 0 ~
        (maintenance_expenses + non_fuel_operation_expenses) / transmission_original_cost,
      is.na((maintenance_expenses + non_fuel_operation_expenses) / transmission_original_cost) ~
        transmission_fixed_OM_frac,
      (maintenance_expenses + non_fuel_operation_expenses) / transmission_original_cost == 0 ~
        transmission_fixed_OM_frac),
    `Transmission_Fixed O&M` = Transmission_CAPEX * transmission_fixed_OM_frac,
    Transmission_Depreciation_Rate = case_when(
      transmission_depreciation_expense / transmission_original_cost > 0 ~
        transmission_depreciation_expense / transmission_original_cost,
      is.na(transmission_depreciation_expense / transmission_original_cost) ~
        transmission_average_depreciation_rate,
      transmission_depreciation_expense / transmission_original_cost == 0 ~
        transmission_average_depreciation_rate)
  )

print("Writing utilities_inputs")
write_parquet(utilities_inputs,"utilities_inputs.parquet")

#group_by(plant_id_eia,boiler_id,report_year) %>%
#mutate(count_units = sum(!is.na(unit_id_eia)), count_gens = sum(!is.na(generator_id))) %>% filter(count_gens>1 & count_units==0)
#%>%
#  select(c(report_year,plant_id_eia,boiler_id,unit_id_eia)) %>%
#  filter(!is.na(unit_id_eia)) %>% distinct() %>% group_by(plant_id_eia,boiler_id,report_year) %>%
#  mutate(count_units = sum(!is.na(unit_id_eia))) %>% filter(count_units>1)
#fuel_splits_and_starts_annual <- fuel_splits_and_starts %>%
#  mutate(Year = year(datetime)) %>%
#  group_by(prime_mover_code,plant_id_eia,generator_id,Year) %>%
#  summarize(
#    energy_source_code_1 = energy_source_code_1,
#    energy_source_code_2 = energy_source_code_2,
#    gross_gen = sum(gross_gen),
#    heat_in_mmbtu = sum(heat_in_mmbtu),
#    co2_tons = sum(co2_tons),
#    fuel_1_mmbtu = sum(fuel_1_mmbtu),
#    fuel_2_mmbtu = sum(fuel_2_mmbtu),
#    gen_starts = sum(gen_starts),
#    fuel_starts = sum(fuel_starts),
#    plant_id_eia = plant_id_eia,
#    generator_id = generator_id,
#    prime_mover_code = prime_mover_code
#  ) %>% distinct() %>% ungroup()
# %>%
#  group_by(report_month, report_year, plant_id_eia, prime_mover_code) %>%
#  mutate(prime_mmbtu = sum(fuel_mmbtu),
#         net_generation_mwh_plant_prime = sum(net_generation_mwh)) %>% ungroup()
#  left_join(gen_monthly_net_mwh, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month"
#  )) %>%
#  left_join(fuel_splits_and_starts %>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_year,
#    report_month,
#    gross_gen
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month"
#  ))  #%>%
#group_by(plant_id_eia,boiler_id,report_year,report_month,boiler_energy_source_code) %>%
#mutate(
#  sum_gen = sum(net_generation_mwh),
#  gen_fuel_mmbtu = ifelse(!(sum_gen==0),net_generation_mwh / sum_gen
#)

#  unique(report_year,report_month,plant_id_eia,boiler_id,boiler_energy_source_code,generator_id)
#  left_join(bga %>% filter(report_year == 2009) %>% rename("generator_id_2009" = "generator_id"), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "boiler_id" = "boiler_id",
#  )) %>%
#  mutate(generator_id  = ifelse(is.na(generator_id),generator_id_2009,boiler_id))

#boiler_mismatch <- boiler_monthly_mmbtu %>% filter(is.na(net_generation_mwh) & boiler_fuel_consumed_mmbtu>0) %>%
#  select(c(plant_id_eia,boiler_id)) %>% distinct()

# read in historical monthly EIA 923 generation and fuel data at the plant-prime-fuel level
#filter(
#  (energy_source_code_hist_1 == boiler_energy_source_code) |
#    (energy_source_code_hist_2 == boiler_energy_source_code) |
#    (energy_source_code_hist_3 == boiler_energy_source_code) |
#    (energy_source_code_hist_4 == boiler_energy_source_code) |
#    (energy_source_code_hist_5 == boiler_energy_source_code) |
#    (energy_source_code_hist_6 == boiler_energy_source_code) |
#    is.na(boiler_energy_source_code)
#) %>%
#filter(
#  (energy_source_code == boiler_energy_source_code) | (
#    (is.na(boiler_energy_source_code) | reported_boilers_not_used_for_gen | is.na(net_generation_mwh)) & (
#      (energy_source_code_hist_1 == energy_source_code) |
#        (energy_source_code_hist_2 == energy_source_code) |
#        (energy_source_code_hist_3 == energy_source_code) |
#        (energy_source_code_hist_4 == energy_source_code) |
#        (energy_source_code_hist_5 == energy_source_code) |
#        (energy_source_code_hist_6 == energy_source_code) |
#        is.na(energy_source_code)
#    )
#  )
#) %>%
#mutate(
#  energy_source_code = ifelse(is.na(energy_source_code),
#                              ifelse(reported_boilers_not_used_for_gen,NA,boiler_energy_source_code),
#                              energy_source_code),
#  energy_source_code_num = case_when(
#    (energy_source_code_hist_1 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_1 == boiler_energy_source_code)) ~ 1,
#    (energy_source_code_hist_2 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_2 == boiler_energy_source_code)) ~ 2,
#    (energy_source_code_hist_3 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_3 == boiler_energy_source_code)) ~ 3,
#    (energy_source_code_hist_4 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_4 == boiler_energy_source_code)) ~ 4,
#    (energy_source_code_hist_5 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_5 == boiler_energy_source_code)) ~ 5,
#    (energy_source_code_hist_6 == energy_source_code) |
#      (is.na(energy_source_code) & !reported_boilers_not_used_for_gen &
#         (energy_source_code_hist_6 == boiler_energy_source_code)) ~ 6,
#    TRUE ~ NA
#  )
#) %>%
#filter(!is.na(energy_source_code)) %>%
#select(-c(
#  energy_source_code_hist_1,
#  energy_source_code_hist_2,
#  energy_source_code_hist_3,
#  energy_source_code_hist_4,
#  energy_source_code_hist_5,
#  energy_source_code_hist_6
#)) %>%
#ppf_num_prime_fuels = sum(!is.na(mmbtu_plant_prime_fuel)),
#ppf_fuel_frac = ifelse(!is.na(ppf_fuel_tot),
#                             ifelse(ppf_fuel_tot>0,mmbtu_plant_prime_fuel / ppf_fuel_tot,(1/ppf_num_prime_fuels)),1),
#net_generation_mwh = (net_generation_mwh * ppf_fuel_frac)
#gen_energy_sources = sum(ifelse(is.na(net_generation_mwh),0,net_generation_mwh))

#%>%
#  group_by(plant_id_eia,prime_mover_code_hist,energy_source_code,report_year,report_month) %>%
#  mutate(
#    ppf_capacity_tot = sum(capacity_hist),
#    ppf_capacity_frac = ifelse(ppf_capacity_tot>0,capacity_hist/ppf_capacity_tot,0),
#    ppf_gen_count = sum(!is.na(net_generation_mwh_plant_prime_fuel)),
#    ppf_with_gen_data_count = sum(!is.na(net_generation_mwh)),
#    ppf_with_boiler_data_count = sum(!is.na(boiler_fuel_consumed_mmbtu)),
#    partial_gen_flag = !(ppf_gen_count==ppf_with_gen_data_count),
#    partial_fuel_flag = !(ppf_gen_count==ppf_with_boiler_data_count)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,generator_id,report_year,report_month) %>%
#  mutate(
#    pg_boiler_fuel_tot = sum(ifelse(!is.na(net_generation_mwh) & !is.na(boiler_fuel_consumed_mmbtu),boiler_fuel_consumed_mmbtu,NA)),
#    pg_boiler_fuel_frac = ifelse(!is.na(pg_boiler_fuel_tot), boiler_fuel_consumed_mmbtu / pg_boiler_fuel_tot,NA),
#    net_generation_mwh = ifelse(is.na(pg_boiler_fuel_frac),net_generation_mwh,net_generation_mwh * pg_boiler_fuel_frac)
#    #gen_energy_sources = sum(ifelse(is.na(net_generation_mwh),0,net_generation_mwh))
#  ) %>% ungroup() %>%


#  group_by(plant_id_eia,boiler_id,report_year,report_month,energy_source_code) %>%
#  mutate(
#    pb_net_gen_tot = sum(ifelse(!is.na(net_generation_mwh) & !is.na(boiler_fuel_consumed_mmbtu),net_generation_mwh,0)),
#    pb_net_gen_frac = ifelse(pb_net_gen_tot>0,net_generation_mwh / pb_net_gen_tot,0),
#    net_generation_mwh = net_generation_mwh * pb_net_gen_frac
#    #gen_energy_sources = sum(ifelse(is.na(net_generation_mwh),0,net_generation_mwh))
#  ) %>% ungroup() %>%
#  group_by(prime_mover_code_hist,report_year)
#group_by(ppf_index,plant_id_eia) %>%
#  mutate(
#    ppf_matching_gens = ifelse(is.na(ppf_index),NA,n()),
#    ppf_matching_reporting_gens = ifelse(is.na(ppf_index),NA,sum(!is.na(net_generation_mwh))),
#    ppf_sum_net_generation_mwh = ifelse(is.na(ppf_index),NA,sum(net_generation_mwh, na.rm = TRUE)),
#    ppf_unalloc_net_generation_mwh = net_generation_mwh_plant_prime_fuel - ppf_sum_net_generation_mwh,
#    ppf_complete = (pbf_matching_gens==pbf_matching_reporting_gens),
#    ppf_primes = ifelse(is.na(bf_index),NA,n_distinct(prime_mover_code_hist, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(ppf_index,plant_id_eia) %>%
#  mutate(
#    ppf_matching_gens = ifelse(is.na(ppf_index),NA,n()),
#    ppf_matching_reporting_gens = ifelse(is.na(ppf_index),NA,sum(!is.na(net_generation_mwh))),
#    ppf_sum_net_generation_mwh = ifelse(is.na(ppf_index),NA,sum(net_generation_mwh, na.rm = TRUE)),
#    ppf_unalloc_net_generation_mwh = net_generation_mwh_plant_prime_fuel - ppf_sum_net_generation_mwh,
#    ppf_complete = (pbf_matching_gens==pbf_matching_reporting_gens),
#    ppf_primes = ifelse(is.na(bf_index),NA,n_distinct(prime_mover_code_hist, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(bf_index,plant_id_eia) %>%
#  mutate(
#    pbf_matching_gens = ifelse(is.na(bf_index),NA,n()),
#    pbf_matching_reporting_gens = ifelse(is.na(bf_index),NA,sum(!is.na(net_generation_mwh))),
#    pbf_complete = (pbf_matching_gens==pbf_matching_reporting_gens),
#    pbf_primes = ifelse(is.na(bf_index),NA,n_distinct(prime_mover_code_hist, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,generator_id,report_year,report_month) %>%
#  mutate(
#    pg_matching_bf = ifelse(is.na(bf_index),NA,sum(!is.na(bf_index))),
#    pg_matching_b = ifelse(is.na(boiler_id),NA,n_distinct(boiler_id, na.rm = TRUE)),
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(
#    max_bf_index = ifelse(is.na(bf_index),NA,max(bf_index, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(bf_index,plant_id_eia) %>%
#  mutate(
#    max_bf_index = ifelse(is.na(max_bf_index),NA,max(max_bf_index, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(
#    max_bf_index = ifelse(is.na(bf_index),NA,max(bf_index, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(bf_index,plant_id_eia) %>%
#  mutate(
#    max_bf_index = ifelse(is.na(max_bf_index),NA,max(max_bf_index, na.rm = TRUE))
#  ) %>% ungroup() %>%
#  group_by(bf_index,plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(pbf_matching_gens_in_ppf = n_distinct(generator_id)) %>% ungroup() %>%
#  group_by(bf_index,plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(pbf_matching_gens_in_pug = sum(!is.na(boiler_id))) %>% ungroup() %>%
#  add_count(boiler_id,plant_id_eia,report_year,report_month, name = "pb_matching_lines") %>%
#  add_count(boiler_id,plant_id_eia,prime_mover_code_hist,report_year,report_month, name = "pb_matching_lines_in_pp") %>%
#  add_count(boiler_id,plant_id_eia,unit_gen_code,report_year,report_month, name = "pb_matching_lines_in_pug") %>%
#  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(
#    complete_data_line = !is.na(boiler_id) & !is.na(net_generation_mwh),
#    pug_reporting_boiler_fuels = sum(complete_data_line),
#    pug_lines = sum(is.na(generator_id)),
#    pug_fuel_mmbtu = sum(ifelse(pb_matching_lines == pb_matching_lines_in_pug,
#                                boiler_fuel_consumed_mmbtu / pbf_matching_gens,NA)),
#    pug_mmbtu_per_mwh = ifelse(pug_net_generation_mwh==0 | is.na(pug_net_generation_mwh),NA, pug_fuel_mmbtu / pug_net_generation_mwh)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,unit_gen_code,energy_source_code,report_year,report_month) %>%
#  mutate(
#    pug_fuel_frac = sum(boiler_fuel_consumed_mmbtu / pbf_matching_gens)/pug_fuel_mmbtu
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,boiler_id,report_year,report_month) %>%
#  mutate(
#    pb_reporting_lines = sum(complete_data_line),
#    pb_fuel_mmbtu = sum(ifelse(pb_matching_lines == pb_reporting_lines & pg_matching_b == 1,
#                               boiler_fuel_consumed_mmbtu / pbf_matching_gens,NA)),
#    pb_net_generation_mwh = sum(ifelse(pb_matching_lines == pb_reporting_lines & pg_matching_b == 1,
#                                       net_generation_mwh / pbf_matching_gens,NA)),
#    pb_mmbtu_per_mwh = ifelse(pb_net_generation_mwh==0 | is.na(pb_net_generation_mwh),NA, pb_fuel_mmbtu / pb_net_generation_mwh)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,boiler_id,energy_source_code,report_year,report_month) %>%
#  mutate(
#    pb_fuel_frac = sum(boiler_fuel_consumed_mmbtu / pbf_matching_gens)/pb_fuel_mmbtu
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(
#    pp_boiler_lines = sum(is.na(generator_id)),
#    pp_reporting_boiler_lines = sum(complete_data_line),
#    pp_boilers_fuel_mmbtu = sum(ifelse(pb_matching_lines == pb_matching_lines_in_pp,
#                                       boiler_fuel_consumed_mmbtu / pbf_matching_gens,0)),
#    pp_boilers_net_generation_mwh = sum(ifelse(pb_matching_lines == pb_matching_lines_in_pp,net_generation_mwh / pug_reporting_boiler_fuels,0)),
#    pp_boilers_mmbtu_per_mwh = ifelse(pp_boilers_net_generation_mwh==0 | is.na(pp_boilers_net_generation_mwh),
#                                      NA, pp_boilers_fuel_mmbtu / pp_boilers_net_generation_mwh),
#    pp_boilers_fuel_mmbtu = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),boiler_fuel_consumed_mmbtu,0)),
#    pp_boilers_mmbtu_per_mwh = ifelse(pp_boilers_net_generation_mwh==0,NA, pp_boilers_fuel_mmbtu / pp_boilers_net_generation_mwh),
#    pp_sum_capacity_hist = sum(capacity_hist),
#    pp_capacity_frac = capacity_hist/ pp_sum_capacity_hist,
#    pp_sum_capacity_hist_no_data = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),0,capacity_hist)),
#    pp_capacity_frac_no_data = ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh) & pp_sum_capacity_hist_no_data>0,
#                                      capacity_hist/ pp_sum_capacity_hist_no_data,NA)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,energy_source_code,report_year,report_month) %>%
#  mutate(
#    ppf_boilers_fuel_mmbtu = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),boiler_fuel_consumed_mmbtu,0)),
#    pp_boilers_fuel_frac =  ifelse(pp_boilers_fuel_mmbtu>0,ppf_boilers_fuel_mmbtu / pp_boilers_fuel_mmbtu,NA)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,generator_id,energy_source_code,report_year,report_month) %>%
#  summarize(
#    unit_code = unit_code,
#    unit_gen_code = unit_gen_code,
#    Prime_hist = Prime_hist,
#    prime_mover_code_hist = prime_mover_code_hist,
#    capacity_hist = capacity_hist,
#    month_hours = month_hours,
#    net_generation_mwh = net_generation_mwh,
#    pug_reporting_gens = pug_reporting_gens,
#    pug_gens = pug_gens,
#    pp_gens = pp_gens,
#    pug_net_generation_mwh = pug_net_generation_mwh,
#    pp_boilers_net_generation_mwh = pp_boilers_net_generation_mwh,
#    pug_sum_capacity_hist = pug_sum_capacity_hist,
#    pp_sum_capacity_hist = pp_sum_capacity_hist,
#    pp_sum_capacity_hist_no_data = pp_sum_capacity_hist_no_data,
#    pug_capacity_frac = pug_capacity_frac,
#    pp_capacity_frac = pp_capacity_frac,
#    pp_capacity_frac_no_data = pp_capacity_frac_no_data,
#    pug_fuel_mmbtu = pug_fuel_mmbtu,
#    pp_boilers_fuel_mmbtu = pp_boilers_fuel_mmbtu,
#    ppf_boilers_fuel_mmbtu = ppf_boilers_fuel_mmbtu,
#    pug_mmbtu_per_mwh = pug_mmbtu_per_mwh,
#    pp_boilers_mmbtu_per_mwh = pp_boilers_mmbtu_per_mwh,
#    pug_fuel_frac = pug_fuel_frac,
#    pp_boilers_fuel_frac = pp_boilers_fuel_frac
#  ) %>% ungroup() %>%
#  left_join(gen_fuel_923, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "prime_mover_code_hist" = "prime_mover_code",
#    "report_year" = "report_year",
#    "report_month" = "report_month",
#    "energy_source_code" = "energy_source_code"
#  )) %>%
#  add_count(ppf_index, name = "ppf_reporting_boilers_gen_fuels") %>%
#  add_count(plant_id_eia,prime_mover_code_hist,report_year,report_month, name = "pp_gen_fuels") %>%
#  group_by(plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(
#    pp_boilers_fuel_mmbtu = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),boiler_fuel_consumed_mmbtu,0)),
#    pp_boilers_net_generation_mwh = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),net_generation_mwh / pug_reporting_boiler_fuels,0)),
#    pp_boilers_mmbtu_per_mwh = ifelse(pp_boilers_net_generation_mwh==0,NA, pp_boilers_fuel_mmbtu / pp_boilers_net_generation_mwh),
#    pp_sum_capacity_hist = sum(capacity_hist),
#    pp_capacity_frac = capacity_hist/ pp_sum_capacity_hist,
#    pp_sum_capacity_hist_no_data = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),0,capacity_hist)),
#    pp_capacity_frac_no_data = ifelse(pp_sum_capacity_hist_no_data>0, capacity_hist/ pp_sum_capacity_hist_no_data,NA),
#    pp_fuel_mmbtu = sum(mmbtu_plant_prime_fuel/ppf_reporting_boilers_gen_fuels)-pp_boilers_fuel_mmbtu,
#    pp_net_generation_mwh = sum(net_generation_mwh_plant_prime_fuel/ppf_reporting_boilers_gen_fuels)-pp_boilers_net_generation_mwh,
#    pp_mmbtu_per_mwh = ifelse(pp_net_generation_mwh==0,NA, pp_fuel_mmbtu / pp_net_generation_mwh),
#    CF = ifelse(is.na(CF),pp_net_generation_mwh /(month_hours * pp_sum_capacity_hist_no_data),CF),
#    mmbtu_per_mwh = case_when(
#      !is.na(pug_mmbtu_per_mwh) ~ pug_mmbtu_per_mwh,
#      !is.na(pp_boilers_mmbtu_per_mwh) ~ pp_boilers_mmbtu_per_mwh,
#      TRUE ~ pp_mmbtu_per_mwh
#    )
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,energy_source_code,report_year,report_month) %>%
#  mutate(
#    ppf_boilers_fuel_mmbtu = sum(ifelse(!is.na(boiler_id) & !is.na(net_generation_mwh),boiler_fuel_consumed_mmbtu,0)),
#    pp_boilers_fuel_frac =  ifelse(pp_boilers_fuel_mmbtu>0,ppf_boilers_fuel_mmbtu / pp_boilers_fuel_mmbtu,NA),
#    ppf_fuel_mmbtu = sum(mmbtu_plant_prime_fuel/ppf_reporting_boilers_gen_fuels)-ppf_boilers_fuel_mmbtu,
#    ppf_fuel_frac = ppf_fuel_mmbtu / pp_fuel_mmbtu,
#    fuel_frac = case_when(
#      !is.na(pug_fuel_frac) ~ pug_fuel_frac,
#      !is.na(pp_boilers_fuel_frac) ~ pp_boilers_fuel_frac,
#      TRUE ~ ppf_fuel_frac
#    )
#  ) %>% ungroup() %>%

#fuel_splits_and_starts <- fuel_splits_and_starts %>%
#  mutate(
#    mmbtu_per_gross_mwh = ifelse(gross_gen>0, heat_in_mmbtu / gross_gen,NA),
#    co2_per_mmbtu = ifelse(heat_in_mmbtu>0,co2_tons / heat_in_mmbtu,NA),
#    co2_per_gross_mwh = ifelse(gross_gen>0,co2_tons / gross_gen,NA),
#  ) %>%
#  left_join(fuel_map, by = c("energy_source_code_1" = "energy_source_code")) %>% rename("fuel_group_code_1" = "fuel_group_code") %>%
#  left_join(fuel_map, by = c("energy_source_code_2" = "energy_source_code")) %>% rename("fuel_group_code_2" = "fuel_group_code")

#ppf_capacity_table <- read_excel("Plant_Prime_Fuel_Capacity_by_Year.xlsx")
#unit_level_data <- read_parquet("../patio_data/unit_level_costs_with_flag.parquet")

#unit_level_data <- unit_level_data %>%
#  rename(
#    "plant_id_eia" = "Plant_ID",
#    "prime_mover_code" = "Prime_with_CCs",
#    "generator_id" = "Generator_ID"
#  ) %>%
#  left_join(all_years, by = character(), copy = TRUE) %>% rename(report_year = y) %>%
#  left_join(all_months, by = character(), copy = TRUE) %>% rename(report_month = y) %>%
#  filter(
#    (is.na(Retirement_Year) |
#       (Retirement_Year > report_year) |
#       ((Retirement_Year==report_year ) & (Retirement_Month > report_month))) &
#      (is.na(Planned_Retirement_Year) |
#         (Planned_Retirement_Year > report_year) |
#         ((Planned_Retirement_Year==report_year ) & (Planned_Retirement_Month > report_month)))) %>%
#  left_join(historic_860_essentials, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year"
#  ))
#left_join(fuel_splits_and_starts %>% select(c(
#  plant_id_eia,
#  generator_id,
#  report_year,
#  report_month,
#  gross_gen,
#  gen_starts,
#  fuel_starts
#)), by = c(
#  "plant_id_eia" = "plant_id_eia",
#  "generator_id" = "generator_id",
#  "report_year" = "report_year",
#  "report_month" = "report_month"
#)) %>%
#  left_join(fuel_splits_and_starts %>% filter(report_year == 2020)%>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_month,
#    mmbtu_per_gross_mwh,
#    co2_per_mmbtu,
#    co2_per_gross_mwh,
#    fuel_1_frac,
#    fuel_2_frac,
#    energy_source_code_1,
#    energy_source_code_2,
#    fuel_group_code_1,
#    fuel_group_code_2
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_month" = "report_month"
#  )) %>%
#  left_join(fuel_splits_and_starts %>% filter(report_year == 2019)%>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_month,
#    mmbtu_per_gross_mwh_2 = mmbtu_per_gross_mwh,
#    co2_per_mmbtu_2 = co2_per_mmbtu,
#    co2_per_gross_mwh_2 = co2_per_gross_mwh,
#    fuel_1_frac_2 = fuel_1_frac,
#    fuel_2_frac_2 = fuel_2_frac,
#    energy_source_code_1_2 = energy_source_code_1,
#    energy_source_code_2_2 = energy_source_code_2,
#    fuel_group_code_1_2 = fuel_group_code_1,
#    fuel_group_code_2_2 = fuel_group_code_2
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_month" = "report_month"
#  )) %>% mutate(
#    mmbtu_per_gross_mwh = ifelse(is.na(mmbtu_per_gross_mwh),mmbtu_per_gross_mwh_2,mmbtu_per_gross_mwh),
#    co2_per_mmbtu = ifelse(is.na(co2_per_mmbtu),co2_per_mmbtu_2,co2_per_mmbtu),
#    co2_per_gross_mwh = ifelse(is.na(co2_per_gross_mwh),co2_per_gross_mwh_2,co2_per_gross_mwh),
#    fuel_1_frac = ifelse(is.na(fuel_1_frac),fuel_1_frac_2,fuel_1_frac),
#    fuel_2_frac = ifelse(is.na(fuel_2_frac),fuel_2_frac_2,fuel_2_frac),
#    energy_source_code_1 = ifelse(is.na(energy_source_code_1),energy_source_code_1_2,energy_source_code_1),
#    energy_source_code_2 = ifelse(is.na(energy_source_code_2),energy_source_code_2_2,energy_source_code_2),
#    fuel_group_code_1 = ifelse(is.na(fuel_group_code_1),fuel_group_code_1_2,fuel_group_code_1),
#    fuel_group_code_2 = ifelse(is.na(fuel_group_code_2),fuel_group_code_2_2,fuel_group_code_2)
#  ) %>% select(-c(
#    mmbtu_per_gross_mwh_2,
#    co2_per_mmbtu_2,
#    co2_per_gross_mwh_2,
#    fuel_1_frac_2,
#    fuel_2_frac_2,
#    energy_source_code_1_2,
#    energy_source_code_2_2,
#    fuel_group_code_1_2,
#    fuel_group_code_2_2
#  )) %>%
#  left_join(fuel_splits_and_starts %>% filter(report_year == 2018)%>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_month,
#    mmbtu_per_gross_mwh_2 = mmbtu_per_gross_mwh,
#    co2_per_mmbtu_2 = co2_per_mmbtu,
#    co2_per_gross_mwh_2 = co2_per_gross_mwh,
#    fuel_1_frac_2 = fuel_1_frac,
#    fuel_2_frac_2 = fuel_2_frac,
#    energy_source_code_1_2 = energy_source_code_1,
#    energy_source_code_2_2 = energy_source_code_2,
#    fuel_group_code_1_2 = fuel_group_code_1,
#    fuel_group_code_2_2 = fuel_group_code_2
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_month" = "report_month"
#  )) %>% mutate(
#    mmbtu_per_gross_mwh = ifelse(is.na(mmbtu_per_gross_mwh),mmbtu_per_gross_mwh_2,mmbtu_per_gross_mwh),
#    co2_per_mmbtu = ifelse(is.na(co2_per_mmbtu),co2_per_mmbtu_2,co2_per_mmbtu),
#    co2_per_gross_mwh = ifelse(is.na(co2_per_gross_mwh),co2_per_gross_mwh_2,co2_per_gross_mwh),
#    fuel_1_frac = ifelse(is.na(fuel_1_frac),fuel_1_frac_2,fuel_1_frac),
#    fuel_2_frac = ifelse(is.na(fuel_2_frac),fuel_2_frac_2,fuel_2_frac),
#    energy_source_code_1 = ifelse(is.na(energy_source_code_1),energy_source_code_1_2,energy_source_code_1),
#    energy_source_code_2 = ifelse(is.na(energy_source_code_2),energy_source_code_2_2,energy_source_code_2),
#    fuel_group_code_1 = ifelse(is.na(fuel_group_code_1),fuel_group_code_1_2,fuel_group_code_1),
#    fuel_group_code_2 = ifelse(is.na(fuel_group_code_2),fuel_group_code_2_2,fuel_group_code_2)
#  ) %>% select(-c(
#    mmbtu_per_gross_mwh_2,
#    co2_per_mmbtu_2,
#    co2_per_gross_mwh_2,
#    fuel_1_frac_2,
#    fuel_2_frac_2,
#    energy_source_code_1_2,
#    energy_source_code_2_2,
#    fuel_group_code_1_2,
#    fuel_group_code_2_2
#  )) %>%
#  left_join(gen_monthly_net_mwh, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month"
#  )) %>%

#  group_by(report_month, report_year, plant_id_eia, prime_mover_code) %>%
#  mutate(
#    mmbtu_pp = sum(mmbtu_923gf),
#    net_generation_mwh_pp = sum(net_generation_mwh_923gf)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, prime_mover_code, energy_source_code) %>%
#  mutate(
#    mmbtu_ppf = sum(mmbtu_923gf),
#    net_generation_mwh_ppf = sum(net_generation_mwh_923gf)
#    fuel_frac_ppf = mmbtu_ppf/mmbtu_pp,
#    gen_frac_ppf = net_generation_mwh_ppf/net_generation_mwh_pp,
#    mmbtu_per_mwh_ppf = mmbtu_ppf/net_generation_mwh_ppf
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, prime_mover_code,fuel_group_code) %>%
#  mutate(
#    mmbtu_ppfg = sum(mmbtu_923gf),
#    net_generation_mwh_ppfg = sum(net_generation_mwh_923gf)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia) %>%
#  mutate(
#    mmbtu_p = sum(mmbtu_923gf),
#    net_generation_mwh_p = sum(net_generation_mwh_923gf)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, energy_source_code) %>%
#  mutate(
#    mmbtu_pf = sum(mmbtu_923gf),
#    net_generation_mwh_pf = sum(net_generation_mwh_923gf)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, fuel_group_code) %>%
#  mutate(
#    mmbtu_pfg = sum(mmbtu_923gf),
#    net_generation_mwh_pfg = sum(net_generation_mwh_923gf)
#  ) %>% ungroup() %>%

#  group_by(report_month, report_year, plant_id_eia, unit_gen_code) %>%
#  mutate(
#    n_ppf = n_distinct(energy_source_code, na.rm = TRUE),
#    n_ppfg = n_distinct(fuel_group_code, na.rm = TRUE)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, ppf_group) %>%
#  mutate(
#    n_unit_gens = n_distinct(unit_gen_code, na.rm = TRUE),
#  ) %>% ungroup()

#boiler_monthly_mmbtu <- boiler_monthly_mmbtu %>%
#  rbind(boiler_monthly_mmbtu %>% mutate(
#    boiler_energy_source_code = NA,
#    mmbtu_bm_pbgfg = NA,
#    mmbtu_bm_pbgf = NA
#  ) %>% distinct())

#  group_by(report_month, report_year, plant_id_eia) %>%
#  mutate(
#    mmbtu_bm_p = sum(mmbtu_bm_pbf)
#  ) %>% ungroup() %>%
#  group_by(report_month, report_year, plant_id_eia, boiler_id) %>%
#  mutate(
#    mmbtu_bm_pb = sum(mmbtu_bm_pbf)
#  ) %>% ungroup() %>%

#  group_by(plant_id_eia,report_year,report_month) %>%
#  mutate(
#    capacity_hist_bg_p = sum(ifelse(!is.na(boiler_group) & (n_plants_bg==1),capacity_hist/n_bgflines_gen,0)),
#    capacity_frac_bg_p = ifelse(!is.na(boiler_group) & (n_plants_bg==1),capacity_hist / capacity_hist_bg_p,NA),
#    capacity_hist_p_res = sum(ifelse(!is.na(boiler_group) & (n_plants_bg==1),0,capacity_hist)),
#    capacity_frac_p_res = ifelse(!is.na(boiler_group) & (n_plants_bg==1),NA,capacity_hist / capacity_hist_p_res)
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(
#    capacity_hist_bg_pp = sum(ifelse(!is.na(boiler_group) & (n_primes_bg==1),capacity_hist/n_bgflines_gen,0)),
#    capacity_frac_bg_pp = ifelse(!is.na(boiler_group) & (n_primes_bg==1),capacity_hist / capacity_hist_bg_pp,NA),
#    capacity_hist_pp_res = sum(ifelse(!is.na(boiler_group) & (n_primes_bg==1),0,capacity_hist)),
#    capacity_frac_pp_res = ifelse(!is.na(boiler_group) & (n_primes_bg==1),NA,capacity_hist / capacity_hist_pp_res)
#  ) %>% ungroup()

# historic_860_923_data <- historic_860_923_data %>%
#  left_join(historic_ppf, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "report_year" = "report_year",
#    "report_month" = "report_month",
#    "prime_mover_code_hist" = "prime_mover_code_hist",
#    "unit_gen_code" = "unit_gen_code"
#  )) %>%
#  group_by(plant_id_eia,report_year,report_month,unit_gen_code,energy_source_code) %>%
#  mutate(
#    boiler_ppf_match = (ifelse(is.na(boiler_energy_source_code) | is.na(energy_source_code),0,
#                              sum(boiler_energy_source_code == energy_source_code))>0)
#  ) %>% ungroup()
#  filter(
#    (boiler_ppf_match & (energy_source_code == boiler_energy_source_code)) |
#      (!boiler_ppf_match & is.na(energy_source_code) & !is.na(boiler_energy_source_code)) |
#      (!boiler_ppf_match & !is.na(energy_source_code) & is.na(boiler_energy_source_code))) %>%
#  mutate(
#    energy_source_code_num = ifelse(!boiler_ppf_match & is.na(energy_source_code),NA,energy_source_code_num),
#    energy_source_code = ifelse(!boiler_ppf_match & is.na(energy_source_code),boiler_energy_source_code,energy_source_code),
#  ) %>% distinct() %>% select(-c(fuel_group_code))


#historic_data <- historic_860_923_data %>%
#  left_join(historic_ppf, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "unit_gen_code" = "unit_gen_code",
#    "report_year" = "report_year",
#    "report_month" = "report_month"
#  )) %>%
#  filter(
#    (boiler_energy_source_included & (energy_source_code == boiler_energy_source_code)) |
#      (!boiler_energy_source_included & is.na(energy_source_code))
#  ) %>%
#  left_join(gen_fuel_923, by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "prime_mover_code_hist" = "prime_mover_code",
#    "report_year" = "report_year",
#    "report_month" = "report_month",
#    "energy_source_code" = "energy_source_code"
#  )) %>%
#  mutate(
#    energy_source_code_num = ifelse(!boiler_energy_source_included & is.na(energy_source_code),NA,energy_source_code_num),
#    energy_source_code = ifelse(!boiler_energy_source_included & is.na(energy_source_code),boiler_energy_source_code,energy_source_code),
#  ) %>% distinct() %>% select(-c(fuel_group_code))

#historic_860_923_data <- historic_860_923_data %>%
#  left_join(fuel_map, by = c("energy_source_code" = "energy_source_code")) %>%
#  group_by(boiler_group,report_year,report_month) %>%
#  mutate(
#    n_lines_pbg = ifelse(is.na(boiler_group),NA,n())
#  ) %>% ungroup() %>%
#  group_by(boiler_group,energy_source_code,report_year,report_month) %>%
#  mutate(
#    n_lines_pbgf = ifelse(is.na(boiler_group),NA,n())
#  ) %>% ungroup() %>%
#  group_by(boiler_group,fuel_group_code,report_year,report_month) %>%
#  mutate(
#    n_lines_pbgfg = ifelse(is.na(boiler_group),NA,n())
#  ) %>% ungroup() %>%
#  group_by(ppf_index) %>%
#  mutate(
#    n_lines_ppf = ifelse(is.na(ppf_index),NA,n()),
#    n_unit_gens_ppf = ifelse(is.na(ppf_index),NA,n_distinct(unit_gen_code))
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(
#  mmbtu_pp_pbg = sum(ifelse(is.na(boiler_group),0,ifelse(n_primes_bg==1,mmbtu_bm_pbg/n_lines_pbg,NA))),
#    net_generation_mwh_pp_pbg = sum(ifelse(is.na(boiler_group),0,ifelse(n_primes_bg==1,net_generation_mwh_bm_pbg/n_lines_pbg,NA))),
#    mmbtu_pp_res = mmbtu_pp - mmbtu_pp_pbg,
#    net_generation_mwh_pp_res = net_generation_mwh_pp - net_generation_mwh_pp_pbg
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,energy_source_code,report_year,report_month) %>%
#  mutate(
#    mmbtu_ppf_pbgf = sum(ifelse(is.na(boiler_group),0,ifelse(n_primes_bg==1,mmbtu_bm_pbgf/n_lines_pbgf,NA))),
#    capacity_hist_ppf = sum(capacity_hist/n()),
#    capacity_frac_ppf = capacity_hist/capacity_hist_ppf,
#    mmbtu_ppf_res = mmbtu_ppf - mmbtu_ppf_pbgf
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,prime_mover_code_hist,fuel_group_code,report_year,report_month) %>%
#  mutate(
#    mmbtu_ppfg_pbgfg = sum(ifelse(is.na(boiler_group),0,ifelse(n_primes_bg==1,mmbtu_bm_pbgfg/n_lines_pbgfg,NA))),
#    capacity_hist_ppfg = sum(capacity_hist/n()),
#    capacity_frac_ppfg = capacity_hist/capacity_hist_ppfg,
#    mmbtu_ppfg_res = mmbtu_ppfg - mmbtu_ppfg_pbgfg
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,report_year,report_month) %>%
#  mutate(
#    mmbtu_p_pbg = sum(ifelse(is.na(boiler_group),0,ifelse(n_plants_bg==1,mmbtu_bm_pbg/n_lines_pbg,NA))),
#    net_generation_mwh_p_pbg = sum(ifelse(is.na(boiler_group),0,ifelse(n_plants_bg==1,net_generation_mwh_bm_pbg/n_lines_pbg,NA))),
#    mmbtu_p_res = mmbtu_p - mmbtu_p_pbg,
#    net_generation_mwh_p_res = net_generation_mwh_p - net_generation_mwh_p_pbg
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,energy_source_code,report_year,report_month) %>%
#  mutate(
#    mmbtu_pf_pbgf = sum(ifelse(is.na(boiler_group),0,ifelse(n_plants_bg==1,mmbtu_bm_pbgf/n_lines_pbgf,NA))),
#    capacity_hist_pf = sum(capacity_hist/n()),
#    capacity_frac_pf = capacity_hist/capacity_hist_pf,
#    mmbtu_pf_res = mmbtu_pf - mmbtu_pf_pbgf
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,fuel_group_code,report_year,report_month) %>%
#  mutate(
#    mmbtu_pfg_pbgfg = sum(ifelse(is.na(boiler_group),0,ifelse(n_plants_bg==1,mmbtu_bm_pbgfg/n_lines_pbgfg,NA))),
#    capacity_hist_pfg = sum(capacity_hist/n()),
#    capacity_frac_pfg = capacity_hist/capacity_hist_pfg,
#    mmbtu_pfg_res = mmbtu_pfg - mmbtu_pfg_pbgfg
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(
#    net_generation_mwh_final = case_when(
#      !is.na(net_generation_mwh) ~ net_generation_mwh,
#      !is.na(net_generation_mwh_bm_pug) ~ net_generation_mwh_bm_pug * capacity_frac_pug,
#      !is.na(net_generation_mwh_ppfg) & (n_fuels_pp==1) ~ net_generation_mwh_ppfg * capacity_frac_ppfg,
#      !is.na(net_generation_mwh_pp_res) ~ net_generation_mwh_pp_res * capacity_frac_pp_res,
#      !is.na(net_generation_mwh_pp) ~ net_generation_mwh_pp * capacity_frac_pp,
#      !is.na(net_generation_mwh_pfg) & (n_fuels_p==1) ~ net_generation_mwh_pfg * capacity_frac_pfg,
#      !is.na(net_generation_mwh_p_res) ~ net_generation_mwh_p_res * capacity_frac_p_res,
#      !is.na(net_generation_mwh_p) ~ net_generation_mwh_p * capacity_frac_p,
#    )
#  ) %>% ungroup() %>%
#  select(-c(boiler_energy_source_code))

#historic_860_923_data <- historic_860_923_data %>%
#  left_join(fuel_splits_and_starts %>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_year,
#    report_month,
#    fuel_starts,
#    gen_starts,
#    heat_in_mmbtu,
#    co2_tons,
#    gross_gen
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month"
#  )) %>%
#  left_join(fuel_splits_and_starts %>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_year,
#    report_month,
#    mmbtu_1 = fuel_1_mmbtu,
#    fuel_1_frac,
#    energy_source_code = energy_source_code_1
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month",
#    "energy_source_code" = "energy_source_code"
#  ))%>%
#  left_join(fuel_splits_and_starts %>% select(c(
#    plant_id_eia,
#    generator_id,
#    report_year,
#    report_month,
#    mmbtu_2 = fuel_2_mmbtu,
#    fuel_2_frac,
#    energy_source_code = energy_source_code_2
#  )), by = c(
#    "plant_id_eia" = "plant_id_eia",
#    "generator_id" = "generator_id",
#    "report_year" = "report_year",
#    "report_month" = "report_month",
#    "energy_source_code" = "energy_source_code"
#  )) %>%
#  mutate(
#    mmbtu_fss = ifelse(is.na(mmbtu_1),0,mmbtu_1)+ifelse(is.na(mmbtu_2),0,mmbtu_2),
#    fuel_frac_fss = ifelse(is.na(fuel_1_frac),0,fuel_1_frac)+ifelse(is.na(fuel_2_frac),0,fuel_2_frac),
#    parasitic_load = ifelse(is.na(gross_gen) | is.na(net_generation_mwh),NA,gross_gen - net_generation_mwh)
#  ) %>% select(-c(mmbtu_1,mmbtu_2,fuel_1_frac,fuel_2_frac)) %>%
#  group_by(plant_id_eia,generator_id,report_year,report_month,energy_source_code) %>%
#  mutate(
#    fss_parasitic_load_lines = sum(!is.na(parasitic_load)),
#    fss_lines = sum(!is.na(gross_gen))
#  ) %>% ungroup() %>%
#  group_by(prime_mover_code_hist,energy_source_code,report_year,report_month) %>%
#  mutate(
#    fss_capacity_hist = sum(ifelse(is.na(parasitic_load) | fss_parasitic_load_lines==0,0,
#                                   capacity_hist/fss_parasitic_load_lines)),
#    parasitic_load_per_MW = ifelse(fss_capacity_hist==0,NA,
#      sum(ifelse(is.na(parasitic_load) | fss_parasitic_load_lines==0,0,
#                 parasitic_load/fss_parasitic_load_lines))/fss_capacity_hist),
#    parasitic_load = ifelse(is.na(parasitic_load),
#                            parasitic_load_per_MW * capacity_hist,
#                            parasitic_load)
#    net_generation_mwh = CF * month_hours,
#    gross_gen = ifelse(is.na(gross_gen) & !is.na(net_generation_mwh) & !is.na(parasitic_load),
#                       net_generation_mwh+parasitic_load,gross_gen),
#    net_generation_mwh = ifelse(!is.na(gross_gen) & is.na(net_generation_mwh) & !is.na(parasitic_load),
#                                gross_gen - parasitic_load, net_generation_mwh),
#    fuel_mmbtu = fuel_frac * mmbtu_per_mwh * net_generation_mwh
#  ) %>% ungroup()

#%>%
#  group_by(plant_id_eia,prime_mover_code_hist,report_year,report_month) %>%
#  mutate(
#    pp_gens = n(),
#    pp_reporting_gens = sum(!is.na(net_generation_mwh)),
#    capacity_hist_pp = sum(capacity_hist),
#    capacity_frac_pp = capacity_hist / capacity_hist_pp
#  ) %>% ungroup() %>%
#  group_by(plant_id_eia,report_year,report_month) %>%
#  mutate(
#    p_gens = n(),
#    p_reporting_gens = sum(!is.na(net_generation_mwh)),
#    capacity_hist_p = sum(capacity_hist),
#    capacity_frac_p = capacity_hist / capacity_hist_p
#  ) %>% ungroup()

#historic_860_923_data <- historic_860_923_data %>%
#  group_by(plant_id_eia,unit_gen_code,report_year,report_month) %>%
#  mutate(
#    net_generation_mwh_final = case_when(
#      !is.na(net_generation_mwh) ~ net_generation_mwh,
#      !is.na(net_generation_mwh_bm_pug) ~ net_generation_mwh_bm_pug * capacity_frac_pug,
#      !is.na(net_generation_mwh_ppfg) & (n_distinct(fuel_group_code)==1) ~ net_generation_mwh_ppfg,
#      !is.na(net_generation_mwh_pp_res) ~ net_generation_mwh_pp_res * capacity_frac_pp,
#      !is.na(net_generation_mwh_pp) ~ net_generation_mwh_pp * capacity_frac_pp,
#      !is.na(net_generation_mwh_p_res) ~ net_generation_mwh_p_res * capacity_frac_p,
#      !is.na(net_generation_mwh_p) ~ net_generation_mwh_p * capacity_frac_p,
#    )
#  ) %>%
#  filter(is.na(net_generation_mwh_final) & is.na(energy_source_code_num)) %>%
#  select(-c(boiler_energy_source_code)) %>%>

#missing_costs <- historic_860_923_data %>%
#  filter(is.na(historic_860_923_data$final_fuel_cost_per_mmbtu) & !(historic_860_923_data$fuel_group_code=="other"))

# Logic for

#  left_join(select(CPIU,c(Year,Inflation_Factor_2020)), by = c("report_year" = "Year")) #%>%
#  left_join(generation_fuel_table,)
#  filter(
#    (is.na(Retirement_Year) |
#       (Retirement_Year > 2022)) &
#      (is.na(Planned_Retirement_Year) |
#         (Planned_Retirement_Year > 2022) ) &
#      !(is.na(fuel_group_code_1) & is.na(fuel_group_code_2)))

capacity_years = c(2020,2025,2030,2035)
for (capacity_year in capacity_years) {
  python_inputs_data <- python_inputs_data %>% left_join(
    ppf_capacity_table %>% filter(report_year == capacity_year) %>%
      subset(select = c(plant_id_eia,prime_mover_code,fuel_code,utility_id_eia,state,capacity_in_report_year)) %>%
      rename_with(~gsub("report_year",as.character(capacity_year),.x, fixed = TRUE)),
    by = c(
      "plant_id_eia" = "plant_id_eia",
      "prime_mover_code" = "prime_mover_code",
      "fuel_code" = "fuel_code",
      "utility_id_eia" = "utility_id_eia",
      "state" = "state"
    )
  )
}

#adjust all extensive data to account for change in plant prime fuel capacity from report date to 2020

python_inputs_data <- python_inputs_data %>%
  mutate(
    capacity_in_2025 = ifelse(capacity_in_2020>0,capacity_in_2025/capacity_in_2020,0),
    capacity_in_2030 = ifelse(capacity_in_2020>0,capacity_in_2030/capacity_in_2020,0),
    capacity_in_2035 = ifelse(capacity_in_2020>0,capacity_in_2035/capacity_in_2020,0),
    capacity_adjustment = ifelse(capacity_in_report_year>0,capacity_in_2020/capacity_in_report_year,0),
    total_fuel_mmbtu = total_fuel_mmbtu * capacity_adjustment,
    total_fossil_mwh = total_fossil_mwh * capacity_adjustment,
    total_carbon = total_carbon * capacity_adjustment,
    total_CO2_emissions = total_CO2_emissions * capacity_adjustment,
    final_fuel_cost = final_fuel_cost * capacity_adjustment,
    capacity_adjusted = capacity_in_2020
  )

python_inputs_data<-python_inputs_data %>%
  group_by(plant_id_eia, prime_mover_code, fuel_code, utility_id_eia, state) %>%
  mutate(
    fossil_CF = ifelse(capacity_adjusted>0,total_fossil_mwh / (capacity_adjusted * ifelse(report_year %% 4 ==0,8784,8760)),0),
    median_CF = median(ifelse(fossil_CF<=1,fossil_CF,NA),na.rm=TRUE),
    high_median_CF = ifelse(median_CF>0.6,1,0),
    mid_median_CF = ifelse(median_CF>0.4 & median_CF<=0.6,1,0),
    low_median_CF = ifelse(median_CF<=0.2,1,0))

python_inputs_data <- python_inputs_data %>%
  left_join(python_inputs_data %>% select(c(plant_id_eia,prime_mover_code,fuel_code,utility_id_eia,state,report_year,total_fossil_mwh,fossil_CF)) %>%
              filter(report_year==2020), by = c(
                "plant_id_eia" = "plant_id_eia",
                "prime_mover_code" = "prime_mover_code",
                "fuel_code" = "fuel_code",
                "utility_id_eia" = "utility_id_eia",
                "state" = "state"
              ), suffix = c("","_2020")) %>% select(-c(report_year_2020))
#python_inputs_data <- python_inputs_data %>%
#  filter(total_fossil_mwh > 0)

python_inputs_data$pollution_control_costs_per_kW<-ifelse(is.na(python_inputs_data$pollution_control_costs_per_kW),0,python_inputs_data$pollution_control_costs_per_kW)

#merge python_inputs_data and reg_variables
python_inputs_data <- merge(python_inputs_data, reg_variables,by.x="prime_mover_code", by.y="Prime")

#merge in FERC data for the current year to use current actuals for FERC reporting plant-primes

FERC_Data_current<-FERC_Data %>% subset(report_year == 2020)

python_inputs_data <- merge(python_inputs_data, FERC_Data_current[,c("Plant", "Prime",
                                                                     "capex_per_kW", "real_opex_per_kW","Fuel 1")],
                            by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                            by.y=c("Plant", "Prime","Fuel 1"), all.x=TRUE)
python_inputs_data <- python_inputs_data %>%
  rename(c(capex_per_kW1 = capex_per_kW,
           real_opex_per_kW1 = real_opex_per_kW))

python_inputs_data <- merge(python_inputs_data, FERC_Data_current[,c("Plant", "Prime",
                                                                     "capex_per_kW", "real_opex_per_kW","Fuel 2")],
                            by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                            by.y=c("Plant", "Prime","Fuel 2"), all.x=TRUE)
python_inputs_data <- python_inputs_data %>%
  rename(c(capex_per_kW2 = capex_per_kW,
           real_opex_per_kW2 = real_opex_per_kW))

python_inputs_data <- merge(python_inputs_data, FERC_Data_current[,c("Plant", "Prime",
                                                                     "capex_per_kW", "real_opex_per_kW","Fuel 3")],
                            by.x=c("plant_id_eia","prime_mover_code","fuel_code"),
                            by.y=c("Plant", "Prime","Fuel 3"), all.x=TRUE)
python_inputs_data <- python_inputs_data %>%
  rename(c(capex_per_kW3 = capex_per_kW,
           real_opex_per_kW3 = real_opex_per_kW))

python_inputs_data <- python_inputs_data %>%
  mutate(capex_per_kW = coalesce(capex_per_kW1,capex_per_kW2,capex_per_kW3),
         real_opex_per_kW = coalesce(real_opex_per_kW1,real_opex_per_kW2,real_opex_per_kW3)) %>%
  select(-c(capex_per_kW1,capex_per_kW2,capex_per_kW3,
            real_opex_per_kW1,real_opex_per_kW2,real_opex_per_kW3))

#add state wage adjustment for current year

wage_factors_recent<-wage_factors %>% subset(year==2020)
python_inputs_data <- merge(python_inputs_data, wage_factors_recent[,c("State","wage_scale")],
                            by.x=c("state"),
                            by.y=c("State"), all.x=TRUE)

#sum(is.na(python_inputs_data$wage_scale))
#sum(is.na(FERC_Data$wage_scale))
#test_nas <- python_inputs_data[is.na(python_inputs_data$wage_scale),]

#calculate estimated capex per kW and annual maintenance capex costs, in nominal dollars

#merge in most recent historical FERC data

#python_inputs_data <- python_inputs_data %>%
#  left_join(FERC_Data %>% filter(report_year == 2020) %>% select(c(
#    Plant,
#    Prime,
#    capex_per_kW,
#    real_opex_per_kW
#  )), by = c(
#    "prime_mover_code" = "Prime",
#    "plant_id_eia" = "Plant"
#  ))

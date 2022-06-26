library(reader)
library(RVenn)
library(stringr)
library(ComplexHeatmap)

# Step 0.1: Start with the Clinical file 
# and find the samples with metastasis & primary tumour

#  change this variable to the cancer type of your choice
Clinical_data <- file("Large_Data_Set/genie_public_clinical_data.tsv", "r")
Clinical_data <- read.csv(Clinical_data, sep = "\t")
closeAllConnections() # Close unintended connections

metastasis_dataset <- subset(Clinical_data, Number.of.Samples.Per.Patient >= 2)

# Make Sure Metastasis_dataset contains only metastasis
patient_id <- unique(metastasis_dataset$Patient.ID)
patient_id_selected <- c()
cancer_type <- "Colorectal Cancer"
cancer_type_concatinated <- "colorectal_cancer"

for(i in 1:length(patient_id)) {
  subset <- subset(metastasis_dataset, Patient.ID == patient_id[i]) 
  sample_type <- subset$Sample.Type
  metastasis_status <- "Metastasis" %in% sample_type
  primary_statis <- "Primary" %in% sample_type
  
  if(metastasis_status && primary_statis) {
    # If this patient has metastasis and primary
    # we choose the patient with primary tumour type == cancer_type
    subset_primary <- subset(subset, Sample.Type == "Primary")
    primary_from_cancer_type <- cancer_type %in% subset_primary$Cancer.Type
    if(primary_from_cancer_type) {
      patient_id_selected <- c(patient_id_selected, patient_id[i])
    }
  }
}

pool_metastasis_patient_id <- patient_id_selected

# Begin subdividing this pool according to metastasis places
metastasis_dataset_cancer_type <- subset(metastasis_dataset, Patient.ID %in% patient_id_selected)
split_metastasis_patient_id_list <- list()

metastasis_type <- c()
for(i in 1 : length(patient_id_selected)) {
  subset <- subset(metastasis_dataset_cancer_type, Patient.ID == patient_id_selected[i])
  subset_metastasis <- subset(subset, Sample.Type == "Metastasis")
  metastasis_cancer_type <- subset_metastasis$Cancer.Type
  for(metastasis_type in metastasis_cancer_type) {
    if(is.na(metastasis_type)) next
    
    if(metastasis_type %in% names(split_metastasis_patient_id_list)) {
      split_metastasis_patient_id_list[[metastasis_type]] <- c(split_metastasis_patient_id_list[[metastasis_type]], patient_id_selected[i])
    } else {
      split_metastasis_patient_id_list[[metastasis_type]] <- c(patient_id_selected[i])
    }
  }
}

## Export the two lists
saveRDS(pool_metastasis_patient_id, file = "pooled_metastasis_patient_id.RDS")
saveRDS(split_metastasis_patient_id_list, file = "split_metastasis_patient_id_list.RDS")

# Step 0.2: Get the gene panel of the study
gene_panels <- file("Large_Data_set/data_gene_matrix.txt", "r")
gene_panels <- read.csv(gene_panels, sep = "\t")
closeAllConnections()
sample_ids <- metastasis_dataset$Sample.ID
gene_panels_cancer_type <- subset(gene_panels, SAMPLE_ID %in% sample_ids)
gene_panels_cancer_type <- unique(gene_panels_cancer_type$mutations) # Please type gene_panels in R console to see what the gene panels are

# Please download the gene panels before running this branch of code
impact341 <- n.readLines("Large_Data_set/data_gene_panel_impact341.txt", n = 1, skip = 2)
impact341 <- strsplit(impact341, split = ":")[[1]]
impact341 <- trimws(impact341[2])
impact341 <- strsplit(impact341, split = "\t")[[1]]

impact410 <- n.readLines("Large_Data_set/data_gene_panel_impact410.txt", n = 1, skip = 2)
impact410 <- strsplit(impact410, split = ":")[[1]]
impact410 <- trimws(impact410[2])
impact410 <- strsplit(impact410, split = "\t")[[1]]

impact468 <- n.readLines("Large_Data_set/data_gene_panel_impact468.txt", n = 1, skip = 2)
impact468 <- strsplit(impact468, split = ":")[[1]]
impact468 <- trimws(impact468[2])
impact468 <- strsplit(impact468, split = "\t")[[1]]

impact505 <- n.readLines("Large_Data_set/data_gene_panel_impact505.txt", n = 1, skip = 2)
impact505 <- strsplit(impact505, split = ":")[[1]]
impact505 <- trimws(impact505[2])
impact505 <- strsplit(impact505, split = "\t")[[1]]

heme_400 <- n.readLines("Large_Data_set/data_gene_panel_heme400.txt", n = 1, skip = 2)
heme_400 <- strsplit(heme_400, split = ":")[[1]]
heme_400 <- trimws(heme_400[2])
heme_400 <- strsplit(heme_400, split = "\t")[[1]]

venn = Venn(list(impact341, impact410, impact468, impact505, heme_400))
gene_list_union <- unite(venn)

dir.create(cancer_type) # create a directory called "Lung"
fileConn<-file(paste0(cancer_type, "/gene_list_union.txt"))
writeLines(gene_list_union, fileConn)
close(fileConn)

# Please take the gene_list_union.txt file and go to
# this website https://www.genenames.org/tools/multi-symbol-checker/
# to get the location of the genes 

# Step 0.3: Extract cancer_type patient from seg file
data_cna_hg19 <- file("Large_Data_Set/genie_data_cna_hg19.seg", "r")
data_cna_hg19_df<- read.csv(data_cna_hg19, sep = "\t")
close(data_cna_hg19)

cencer_type_cna_hg_19 <- subset(data_cna_hg19_df, ID %in% metastasis_dataset_cancer_type$Sample.ID)

file_directory = paste0(cancer_type, "/data_cna_hg19_", cancer_type_concatinated,".seg")
write.table(cencer_type_cna_hg_19, file = file_directory, sep = "\t", row.names = FALSE, quote = FALSE)

# Please take the data_cna_hg19_cancer_type_concatinated.seg 
# to GISTIC 2 and return for the next step
# Step 1: Get the top 40% by Q Value

step_1_get_top_40_by_Q_value <- function (cancer_type){
  selection_threshold = 0.55
  
  all_lesions_path <- paste(cancer_type, "/all_lesions.conf_99.txt", sep = "")
  flag <- T # a flag indicating if this function finishes successfully
  
  if(!file.exists(all_lesions_path)) {
    writeLines(paste(cancer_type, "does not have all_leions_file"))
    Missing_cancer_type <- cancer_type
    flag <- F
    return_list <- list(flag, Missing_cacner_type, NaN, NaN, NaN, NaN)
    return(flag, Missing_cancer_type)
  }
  
  all_lesions.conf_99 <- file(all_lesions_path, "r")
  
  first_line = readLines(all_lesions.conf_99, n = 1) # discard the first line
  first_line_split <- strsplit(first_line, "\t")
  number_of_samples <- length(first_line_split[[1]]) - 9 # samples only start on the 10th column
  
  peak_q_values <- c(); delete_q_values <- c();
  peak_names <- c(); delete_names <- c();peak_cytoband <- c(); delete_cytoband <- c()
  
  # read line by line 
  while(TRUE) {
    line = readLines(all_lesions.conf_99, n = 1)
    line_split <- strsplit(line, "\t")
    if (line_split[[1]][9] == "Actual Copy Change Given" ) {
      break
    } # we only need the data from section 1 (for sections 
    # please refer to manual of GISTIC 2 from GenePattern)
    
    amp_or_del <- grepl("Amplification", line_split[[1]][1])
    if(amp_or_del) {
      peak_q_values <- c(peak_q_values, as.numeric(line_split[[1]][6])) # the 6th column is the q value
      peak_names <- c(peak_names, line_split[[1]][1])
      peak_cytoband <- c(peak_cytoband, line_split[[1]][2])
    } else {
      delete_q_values <- c(delete_q_values, as.numeric(line_split[[1]][6])) # the 6th column is the q value
      delete_names <- c(delete_names, line_split[[1]][1])
      delete_cytoband <- c(delete_cytoband, line_split[[1]][2])
    }
  }
  
  peak_q_values <- as.data.frame(peak_q_values)
  peak_q_values <- cbind(peak_q_values, peak_names)
  peak_q_values <- cbind(peak_q_values, peak_cytoband)
  
  delete_q_values <- as.data.frame(delete_q_values)
  delete_q_values <- cbind(delete_q_values, delete_names)
  delete_q_values <- cbind(delete_q_values, delete_cytoband)
  
  close(all_lesions.conf_99)
  
  
  peak_nrow <- nrow(peak_q_values); delete_nrow <- nrow(delete_q_values);
  sum <- peak_nrow + delete_nrow; after_selection = floor(sum * selection_threshold)
  
  peak_top <- floor(after_selection * peak_nrow / sum); delete_top <- after_selection - peak_top;
  
  peak_q_values_order <- peak_q_values[order(peak_q_values$peak_q_values), ]
  peak_q_values_top <- peak_q_values_order[1:peak_top, ]
  peak_amplification_events <- peak_q_values_top$peak_names 
  
  delete_q_values_order <- delete_q_values[order(delete_q_values$delete_q_values), ]
  delete_q_values_top <- delete_q_values_order[1 : delete_top, ]
  delete_events <- delete_q_values_top$delete_name
  
  return_list <- list(flag, cancer_type, peak_q_values_top, delete_q_values_top, number_of_samples, first_line_split)
  return(return_list)
}

return_list <- step_1_get_top_40_by_Q_value(cancer_type = cancer_type)

# Step 2: Binarised genome
gene_union_HGNC_annotation <- read.csv("Large_Data_Set/hgnc-symbol-check.csv")

# A function to extract genes at a cytoband in "gene_union_HGNC_annotation"
get_genes_at_cytoband <- function(cytoband) {
  genes_in_cytoband <- gene_union_HGNC_annotation$Input[gene_union_HGNC_annotation$Location == cytoband]
  return(genes_in_cytoband)
}

step_2_binarised_genome_construction <- function(return_list_from_step_1){
  peak_q_values_top <- return_list_from_step_1[[3]]
  delete_q_values_top <- return_list_from_step_1[[4]]
  number_of_samples <- return_list_from_step_1[[5]]
  first_line_split <- return_list_from_step_1[[6]]
  cancer_type <- return_list_from_step_1[[2]]
  all_lesions_path <- paste(cancer_type, "/all_lesions.conf_99.txt", sep = "")
  
  genotype_amplification <- data.frame(matrix(ncol = number_of_samples,nrow = 0))
  genotype_deletion <- data.frame(matrix(ncol = number_of_samples, nrow = 0))
  colnames(genotype_amplification) = colnames(genotype_deletion) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
  
  all_lesions.conf_99 = file(all_lesions_path, "r")
  counter_amplification <- 0
  missing_amplification <- c()
  counter_deletion <- 0
  missing_deletion <- c()
  
  while(TRUE) {
    line = readLines(all_lesions.conf_99, n = 1)
    
    if(length(line) == 0) {
      break
    }
    
    line_split <- strsplit(line, "\t")
    if (line_split[[1]][9] != "Actual Copy Change Given" ) {
      next
    }
    
    amp_or_del <- grepl("Amplification", line_split[[1]][1])
    if(amp_or_del) {
      # only use top 40% 
      if(!(str_trim(line_split[[1]][2]) %in% str_trim(peak_q_values_top$peak_cytoband))) {
        next
      }
      
      # Amplification
      cytoband <- str_trim(line_split[[1]][2])
      gene_list <- get_genes_at_cytoband(cytoband)
      number_of_genes <- length(gene_list)
      
      if (number_of_genes == 0) {
        counter_amplification <- counter_amplification + 1
        missing_amplification <- c(missing_amplification, cytoband)
        next
      }
      
      genotype_per_cytoband <- data.frame(matrix(nrow = number_of_genes, ncol = number_of_samples))
      colnames(genotype_per_cytoband) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
      rownames(genotype_per_cytoband) <- make.names(gene_list, unique=TRUE)
      i <- 1
      for(actual_copy_number in line_split[[1]][c(10:(10 + number_of_samples-1))]) {
        if (as.numeric(actual_copy_number) > 0.3) {
          genotype_per_cytoband[, i] <- rep(1, number_of_genes)
          i <- i + 1
        } else {
          genotype_per_cytoband[, i] <- rep(0, number_of_genes)
          i <- i + 1
        }
      }
      
      # Deal with identical gene names in different cytoband
      intersect_genes <- intersect(rownames(genotype_amplification), rownames(genotype_per_cytoband))
      if(length(intersect_genes) == 0){
        genotype_amplification <- rbind(genotype_amplification, genotype_per_cytoband)
      } else {
        for (gene in intersect_genes) {
          genotype_amplification[gene, ] <- as.integer(genotype_amplification[gene, ] | genotype_per_cytoband[gene, ])
        }
      }
    } else {
      if (!(str_trim(line_split[[1]][2]) %in% str_trim(delete_q_values_top$delete_cytoband))) {
        next
      }
      
      # Deletion
      cytoband <- str_trim(line_split[[1]][2])
      gene_list <- get_genes_at_cytoband(cytoband)
      gene_list <- unique(gene_list)
      number_of_genes <- length(gene_list)
      if (number_of_genes == 0) {
        counter_deletion <- counter_deletion + 1
        missing_deletion <- c(missing_deletion, cytoband)
        next
      }
      genotype_per_cytoband <- data.frame(matrix(nrow = number_of_genes, ncol = number_of_samples), check.names = FALSE)
      colnames(genotype_per_cytoband) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
      rownames(genotype_per_cytoband) <- make.names(gene_list, unique=TRUE)
      
      i <- 1
      for(actual_copy_number in line_split[[1]][c(10:(10 + number_of_samples -1))]) {
        if (as.numeric(actual_copy_number) < -0.3) {
          genotype_per_cytoband[, i] <- rep(1, number_of_genes)
          i <- i + 1
        } else {
          genotype_per_cytoband[, i] <- rep(0, number_of_genes)
          i <- i + 1
        }
      }
      
      # Deal with identical gene names in different cytoband
      intersect_genes <- intersect(rownames(genotype_deletion), rownames(genotype_per_cytoband))
      if(length(intersect_genes) == 0){
        genotype_deletion <- rbind(genotype_deletion, genotype_per_cytoband)
      } else {
        for (gene in intersect_genes) {
          genotype_deletion[gene, ] <- as.integer(genotype_deletion[gene, ] | genotype_per_cytoband[gene, ])
        }
      }
    }
  } 
  close(all_lesions.conf_99)
  
  snv_file <- file("Large_Data_Set/data_mutations_extended.txt", "r")
  line = readLines(snv_file, n = 1)
  gene_pool <- c() # get all the snv related genes
  sample_pool <- first_line_split[[1]][10:length(first_line_split[[1]])]
  
  snv_df <- data.frame(matrix(0, nrow = 0, ncol = length(sample_pool)))
  colnames(snv_df) <- sample_pool
  
  while(TRUE) {
    line = readLines(snv_file, n = 1)
    if (length(line) == 0) {
      break
    }
    line_split = strsplit(line, "\t")[[1]]
    tumor_barcode <- line_split[17]; Hugo_Symbol <- line_split[1];
    if(tumor_barcode %in% sample_pool) {
      gene_pool <- c(gene_pool, Hugo_Symbol)
      
      if(Hugo_Symbol %in% rownames(snv_df)) {
        snv_df[Hugo_Symbol, tumor_barcode] <- 1
      } else {
        new_gene <- data.frame(matrix(0, nrow = 1, ncol = length(sample_pool)))
        rownames(new_gene) <- Hugo_Symbol; colnames(new_gene) <- sample_pool 
        snv_df = rbind(snv_df, new_gene)
      }
    }
  }
  gene_pool <- unique(gene_pool)
  close(snv_file)
  
  amplification_genes <- rownames(genotype_amplification)
  deletion_genes <- rownames(genotype_deletion)
  
  intersection_amp_del <- intersect(amplification_genes, deletion_genes)
  genotype_deletion_2 <- genotype_deletion
  for(gene in intersection_amp_del) {
    amp_gene <- genotype_amplification[gene, ]
    del_gene <- genotype_deletion[gene, ]
    amp_del_or <- as.numeric(amp_gene | del_gene)
    genotype_amplification[gene, ] <- amp_del_or
    
    genotype_deletion_2 <- genotype_deletion_2[!row.names(genotype_deletion_2) %in% gene, ]
  }
  
  input_of_Meta_MHN <- rbind(genotype_amplification, genotype_deletion_2)
  row_name <- rownames(input_of_Meta_MHN)
  intersection_input_gene_pool <- intersect(row_name, gene_pool)
  snv_df_2 <- snv_df
  
  for(gene in intersection_input_gene_pool) {
    input_gene <- input_of_Meta_MHN[gene, ]
    snv_gene <- snv_df[gene, ]
    or <- as.numeric(input_gene | snv_gene)
    input_of_Meta_MHN[gene, ] <- or
    snv_df_2 <- snv_df_2[!row.names(snv_df_2) %in% gene, ]
  }
  
  input_of_Meta_MHN <- rbind(input_of_Meta_MHN, snv_df_2)
  input_of_Meta_MHN <- input_of_Meta_MHN[order(rowSums(input_of_Meta_MHN), decreasing = T), ]
  
  # saveRDS(input_of_Meta_MHN, file = paste0(cancer_type, "/Input_of_Meta_MHN.RDS"))
  
  return_list <- list(genotype_amplification, genotype_deletion, snv_df, input_of_Meta_MHN, gene_pool, cancer_type, number_of_samples, first_line_split)
  return(return_list)
}

return_list <- step_2_binarised_genome_construction(return_list)

# Step 3 : Plot top 100 oncoplot
step_3_oncoplots <- function(return_list_from_step_2) {
  genotype_amplification <- return_list_from_step_2[[1]]
  genotype_deletion <- return_list_from_step_2[[2]]
  snv_df <- return_list_from_step_2[[3]]
  gene_pool <- return_list_from_step_2[[5]]
  cancer_type <- return_list_from_step_2[[6]]
  number_of_samples <- return_list_from_step_2[[7]]
  first_line_split <- return_list_from_step_2[[8]]
  
  amplification_genes <- rownames(genotype_amplification)
  deletion_genes <- rownames(genotype_deletion)
  genes_union <- unique(union(amplification_genes, deletion_genes))
  genes_union <- unique(union(genes_union, gene_pool))
  
  missing_from_deletion = setdiff(genes_union, deletion_genes)
  genotype_deletion_oncoplot <- genotype_deletion
  if(length(missing_from_deletion) != 0) {
    to_be_add_to_deletion = data.frame(matrix(0, length(missing_from_deletion), number_of_samples))
    colnames(to_be_add_to_deletion) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
    rownames(to_be_add_to_deletion) <- missing_from_deletion
    genotype_deletion_oncoplot <- rbind(genotype_deletion_oncoplot, to_be_add_to_deletion)
  }
  
  
  missing_from_amplification = setdiff(genes_union, amplification_genes)
  genotype_amplification_oncoplot <- genotype_amplification
  if(length(missing_from_amplification) != 0) {
    to_be_add_to_amplification = data.frame(matrix(0, length(missing_from_amplification), number_of_samples))
    colnames(to_be_add_to_amplification) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
    rownames(to_be_add_to_amplification) <- missing_from_amplification
    genotype_amplification_oncoplot <- rbind(genotype_amplification_oncoplot, to_be_add_to_amplification)
  }
  
  missing_from_snv = setdiff(genes_union, gene_pool)
  genotype_snv_oncoplot <- snv_df
  if(length(missing_from_snv) != 0) {
    to_be_add_to_snv = data.frame(matrix(0, length(missing_from_snv), number_of_samples))
    colnames(to_be_add_to_snv) <- first_line_split[[1]][c(10 : (10 + number_of_samples - 1))]
    rownames(to_be_add_to_snv) <- missing_from_snv
    genotype_snv_oncoplot <- rbind(genotype_snv_oncoplot, to_be_add_to_snv)
  }
  
  # Reorder each oncoplot data frame so that they are in the same order
  genes <- rownames(genotype_amplification_oncoplot)
  genotype_deletion_oncoplot <- genotype_deletion_oncoplot[genes, ]
  genotype_snv_oncoplot <- genotype_snv_oncoplot[genes, ]
  
  # only show top "selection" genes
  genes <- colnames(genotype_amplification_oncoplot)
  row_sum_amplification <- rowSums(genotype_amplification_oncoplot)
  row_sum_deletion <- rowSums(genotype_deletion_oncoplot)
  row_sum_snv <- rowSums(genotype_snv_oncoplot)
  gene_sum <- row_sum_amplification + row_sum_deletion + row_sum_snv
  order <- order(gene_sum, decreasing = TRUE)
  selection <- nrow(genotype_amplification_oncoplot)
  selection <- min(selection, 100)
  top_100_genes <- names(gene_sum)[order[1:selection]]
  
  metastasis_genotype_amplification_100 <- genotype_amplification_oncoplot[top_100_genes, ]
  metastasis_genotype_deletion_100 <- genotype_deletion_oncoplot[top_100_genes, ]
  metastasis_genotype_snv_100 <- genotype_snv_oncoplot[top_100_genes, ]
  
  onco_matrix_list <- list(amp = data.matrix(metastasis_genotype_amplification_100), 
                           del = data.matrix(metastasis_genotype_deletion_100), 
                           snv = data.matrix(metastasis_genotype_snv_100))
  
  col <- c("amp" = "red", "del" = "blue", "snv" = "green")
  alter_fun = list(
    background = alter_graphic("rect", fill = "#CCCCCC"),
    amp = alter_graphic("rect", fill = col["amp"]),
    del = alter_graphic("rect", fill = col["del"]),
    snv = alter_graphic("rect", fill = col["snv"])
  )
  
  column_title = paste(cancer_type, "cancer patients with metastasis top ", selection)
  heatmap_legend_param = list(title = "Alterations", at=c("amp", "del", "snv"), labels = c("Amplification", "Deletion", "snv"))
  png(filename = paste(cancer_type, "/oncoplot_metastasis_top_",selection,".png", sep = ""),width=25,height=50,units="cm",res=1200)
  # png(filename = paste("default_gistic/", cancer_type, "/oncoplot_metastasis_top_",selection,".png", sep = ""),width=25,height=50,units="cm",res=1200)
  print(oncoPrint(onco_matrix_list, alter_fun = alter_fun, col = col, 
                  column_title = column_title, heatmap_legend_param = heatmap_legend_param,
                  height = unit(35, "cm"), width = unit(15, "cm")))
  dev.off()
}

step_3_oncoplots(return_list)

# Some patient's metatasis sample is not present in the seg file
# We need to trim out those patient in out Input_for_Meta_MHN

Input_of_Meta_MHN_trim <- data.frame(matrix(nrow = nrow(Input_of_Meta_MHN), ncol = 0))
rownames(Input_of_Meta_MHN_trim) <- rownames(Input_of_Meta_MHN)
colnames <- c()
for(patient_id in pool_metastasis_patient_id) {
  patient_sample <- subset(metastasis_dataset_cancer_type, Patient.ID == patient_id)
  patient_primary <- subset(patient_sample, Sample.Type == "Primary")
  patient_primary_cancer_type <- subset(patient_primary, Cancer.Type == cancer_type)
  
  if(nrow(patient_primary_cancer_type) > 1) {
    next # discard patients with multiple primary tumor
  }
  
  patient_primary_cancer_type_sample_id <- patient_primary_cancer_type$Sample.ID
  is_primary_in_input_of_meta_MHN <- patient_primary_cancer_type_sample_id %in% colnames(Input_of_Meta_MHN)
  
  
  patient_metastasis <- subset(patient_sample, Sample.Type == "Metastasis")
  if(nrow(patient_metastasis) > 1) {
    ages <- patient_metastasis$Age.at.Which.Sequencing.was.Reported
    age_min <- min(ages)
    patient_metastasis <- subset(patient_metastasis, Age.at.Which.Sequencing.was.Reported == age_min)
  }
  patient_metastasis_sample_id <- patient_metastasis[1, ]$Sample.ID
  is_metastasis_in_input_of_meta_MHN <- patient_metastasis_sample_id %in% colnames(Input_of_Meta_MHN)
  
  # if both metastasis and primary is in the input,
  # we keep this patient
  if(is_primary_in_input_of_meta_MHN && is_metastasis_in_input_of_meta_MHN) {
    primary <- Input_of_Meta_MHN[, patient_primary_cancer_type_sample_id]
    metastasis <- Input_of_Meta_MHN[, patient_metastasis_sample_id]
    Input_of_Meta_MHN_trim <- cbind(Input_of_Meta_MHN_trim, primary)
    Input_of_Meta_MHN_trim <- cbind(Input_of_Meta_MHN_trim, metastasis)
    colnames <- c(patient_primary_cancer_type_sample_id, patient_metastasis_sample_id, colnames)
  }
}

colnames(Input_of_Meta_MHN_trim) <- colnames
Input_of_Meta_MHN_trim <- Input_of_Meta_MHN_trim[order(rowSums(Input_of_Meta_MHN_trim), decreasing = T), ]
saveRDS(Input_of_Meta_MHN_trim, file = paste0(cancer_type, "/Input_of_Meta_MHN_genes.RDS"))

# Adding biological insite to the model 
# According to Which Allelic Loss to Choose.Rmd, 
# we selected four allelic locations.

allelic_locations <- c("18q21.33", "18q12.3",  "17p12",    "17p13.1")

cancer_type <- "Colorectal Cancer"
all_lesions_path <- paste(cancer_type, "/all_lesions.conf_99.txt", sep = "")
all_lesions.conf_99 <- file(all_lesions_path, "r")
all_lesions.conf_99 <- read.csv(all_lesions.conf_99, header = TRUE, sep = "\t")
all_lesions.conf_99 <- subset(all_lesions.conf_99, Amplitude.Threshold == "0: t>-0.3; 1: -0.3>t> -1.3; 2: t< -1.3")
deletions_in_paper <- subset(all_lesions.conf_99, str_trim(Descriptor) %in% allelic_locations)

trimed_Samples <- colnames(Input_of_Meta_MHN_trim)
Input_of_Meta_MHN_bio <- data.frame(matrix(nrow = 4, ncol = 0))
rownames(Input_of_Meta_MHN_bio) <- deletions_in_paper$Descriptor

for(sample_name in trimed_Samples) {
  sample_name <- gsub("-", ".", sample_name)
  Input_of_Meta_MHN_bio <- cbind(Input_of_Meta_MHN_bio, deletions_in_paper[, sample_name])
}

colnames(Input_of_Meta_MHN_bio) <- trimed_Samples
saveRDS(Input_of_Meta_MHN_bio, file = paste0(cancer_type,"/Input_of_Meta_MHN_bio.RDS"))

# Now we can turn give the two Input_of_Meta_MHN_bio.RDS and Input_of_Meta_MHN_genes.RDS to MetaMHN!
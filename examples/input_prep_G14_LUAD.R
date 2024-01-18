
## This script prepares input data for cancer progression modeling with
## metMHN as described in xxx. It is mainly based on "raw" data from
## Memorial Sloan Kettering Cancer Center samples, retrieved through
## AACR GENIE on synapse.org - data release 14.1 - https://doi.org/10.7303/syn52918985

## This script requires the following files from GENIE 14.1:
## - data_clinical_patient.txt
## - data_clinical_sample.txt
## - data_mutations_extended.txt
## - data_cna_hg19.seg
## - data_sv.txt
## - genomic_information.txt
## - gene_panels folder

## Additional files required:
## - data_clinical_patient.txt for the MSK-MET cohort, obtainable at https://www.cbioportal.org/study/summary?id=msk_met_2021
## - data_clinical_sample.txt for the MSK-MET cohort, obtainable at https://www.cbioportal.org/study/summary?id=msk_met_2021
## - chromosome band information for hg19, obtainable at https://hgdownload.soe.ucsc.edu/downloads.html
## - COSMIC's cancer gene census list, obtainable at https://cancer.sanger.ac.uk/census
##   We used v99 (Dec 2023). This file is not required but helps with identifying
##   CNA target regions

## After downloading these files, 3 steps have to be carried out to pre-process
## the GENIE mutation and CNA data
## 1. Annotate data_mutations_extended.txt with OncoKB - see https://github.com/oncokb/oncokb-annotator
##    and DOI: 10.1200/PO.17.00011. Some reformatting may be necessary
## 2. Carry out the filtering strategy described in xxx to get a boolean
##    label for each entry in data_mutations_extended.txt
## 3. Normalise segmented copy number data in data_cna_hg19.seg using
##    mecan4cna - see https://github.com/baudisgroup/mecan4cna/ and
##    DOI: 10.1016/j.ygeno.2020.05.008.
##    Note that mecan4cna will not produce output for CN-silent samples

## This analysis was performed in R 4.3.1

##### CONTENTS

################################################################################
### 01: DATA AND PACKAGE LOADING
### ============================================================================
### load packages and necessary input files as described above. 
### some initial reformatting steps.
################################################################################

################################################################################
### 02: SAMPLE SELECTION
### ============================================================================
### for metMHN, sample selection is carried out as follows: 
### first, only take samples that are annotated with cancer type of interest.
### take only one sample per patient, except take one primary and one metastasis
### from paired samples. if a sample has either multiple prim and or multiple
### met samples, pick the sample with the lowest age at sampling for prims
### and the sample with highest age at sampling for mets.
### always avoid NAs in sampling age. 
### additionally, unpaired primary samples are annotated with their metastatic 
### status: if, and only if, a patient is included in the MSK-MET study and in  
### that data has been assigned a met count of 0, declare the sample metastatic 
### status as "absent" (at time of sampling). if the patient has either 
### a sequenced metastatic sample in GENIE or a met count of > 0 in the MSK-MET
### study, declare the metastatic status as "present". If there is no sequenced
### metastatic sample but the patient is not included in MSK-MET, declare the
### status as "unknown" (there may be a met that has not been sequenced). 
################################################################################

################################################################################
### 03: CNA EVENT DEFINITIONS AND CALLING
### ============================================================================
### THIS SECTION HAS TO BE CARRIED OUT MANUALLY!
### to get CN events, we first consider calibrated LRR values obtained through
### mecan4cna. these have been converted to pseudo-absolute copy-numbers and
### rounded to the nearest integer so that there are 5 levels: 0 (deep del),
### 1 (shallow del), 2 (normal), 3 (moderate gain), 4+ (high-level gain).
### note that these levels are relative. sample gains (3/4+) and deletions 
### (0/1) are mapped to a coordinate list considering all breakpoints found in
### MSK data. this list is then manually evaluated to define one genomic
### interval of greatest interest per chromosome per CNA type (amp/del).
### for this evaluation, usually the minimal common region of a chromosome
### is considered, unless there is a significant biological signal in an area
### covered by a gene which is also a known cancer driver.
### when defining the exact intervals for events, gains are usually required
### to cover entire genes unless the signal clearly suggests something different
### while deletions may only cover a portion of the suspected target gene.
### after the manual definitions, events are called across samples.
### for samples which do not have a mecan4cna output, the script checks that 
### the original data suggest that these are CN-silent, and call no events.
################################################################################

################################################################################
### 04: MUTATION EVENT CALLING
### ============================================================================
### firstly, to avoid including events that were not consistently assayed
### across the different versions of MSK-IMPACT, panel details are used to 
### pre-select genes that are assayed in all MSK-IMPACT versions.
### individual mutations are then filtered by binary functionality/pathogenicity
### annotation. then, any sample which has at least one functional variant
### in a gene of interest will get the respective event assigned.
################################################################################

################################################################################
### 05: FUSION EVENT PREPARATION
### ============================================================================
### THIS SECTION HAS TO BE CARRIED OUT MANUALLY!
### in this section, GENIE SV data is used to manually define distinct fusion
### events which then get binary calls for each sample
################################################################################

################################################################################
### 05: EVENT SELECTION AND CONVERSION TO P/M FORMAT
### ============================================================================
### THIS SECTION HAS TO BE CARRIED OUT MANUALLY!
### there is a limit to active-event-per-sample distributions at which metMHN
### training becomes computationally unfeasible. this limit chiefly depends
### on the maximum number of active (i.e., present) events per observation.
### as of december 2023, our decision rule was therefore that "no complete 
### observation (i.e., primary events + metastasis events + seeding) may
### exceed 20 active events + seeding."
### in order to generate an input satisfying this rule while containing a 
### maximum of interesting events, in this section one first chooses an 
### event set, converts the vanilla input obtained that way to the P/M format
### and then checks whether the rule is satisfied, with an option to omit
### a small fraction of samples which exceed the limit. iteratively, different
### event sets should be tried out.
################################################################################



##### 01: DATA AND PACKAGE LOADING
################################################################################

rm(list=ls())
`%notin%` <- Negate(`%in%`)

# packages
library(GenomicRanges)
library(ComplexHeatmap)

# set directories
baseDir <- paste0(getwd(), "/")
outDir <- paste0(baseDir, "outputs/eventProjects/")

# GENIE metadata
sampleData <- read.delim(paste0(baseDir, "data/data_clinical_sample.txt"), comment.char="#")
genieMetPatients <- unique(sampleData[which(sampleData$SAMPLE_TYPE == "Metastasis"), "PATIENT_ID"])

# load MSK-MET data for metastatic status query
metPatDat <- read.delim(paste0(baseDir, "data/MSK_MET/data_clinical_patient.txt"), comment.char = "#")
metPatDat[, c("AGE_AT_SURGERY", "AGE_AT_DEATH", "AGE_AT_LAST_CONTACT")] <- lapply(metPatDat[, c("AGE_AT_SURGERY", "AGE_AT_DEATH", "AGE_AT_LAST_CONTACT")] , as.numeric)
metSampleDat <- read.delim(paste0(baseDir, "data/MSK_MET/data_clinical_sample.txt"), comment.char = "#")
metSampleDat$MET_COUNT <- as.numeric(metSampleDat$MET_COUNT)

# load annotated mutation file
muts <- read.delim(paste0("outputs/oncokb/data_mutations_MSK_eventFilter.txt"))

# load pre-selected and mecan4cna-calibrated CN data
seg <- read.table(paste0(baseDir, "outputs/mecan4cna/segGENIE14OriginalMecan.txt"), header = T)
# .. and the original CN data
originalSegMSK <- read.delim("data/data_cna_hg19.seg")

# cancer genes (obtained from cosmic V96, June 2022) mapped to hg19
geneLoci <- read.delim(paste0(baseDir, "data/misc/cosmicGeneCensusv99.tsv"))
geneLoci$chromosome_name <- sapply(strsplit(geneLoci$Genome.Location, ":"), function(x) x[1])
geneLoci$start_position <- sapply(strsplit(sapply(strsplit(geneLoci$Genome.Location, "-"), function(x) x[1]), ":"), function(x) x[2])   
geneLoci$end_position <- sapply(strsplit(geneLoci$Genome.Location, "-"), function(x) x[2])
geneLoci <- geneLoci[which(!is.na(geneLoci$start_position)), ]
geneLoci <- geneLoci[which(!is.na(geneLoci$end_position)), ]
glGR <- GRanges(seqnames = geneLoci$chromosome_name, ranges = IRanges(start = as.numeric(geneLoci$start_position), end = as.numeric(geneLoci$end_position)))


# load panel details for TMB calculation
panelProbes <- read.delim(paste0(baseDir, "data/genomic_information.txt"))
# for all msk assays, sum up extent of all covered exonic regions and return in Mb units
coveredMb <- vector(length = 6); names(coveredMb) <- c("MSK-IMPACT-HEME-400", "MSK-IMPACT-HEME-468", "MSK-IMPACT341", "MSK-IMPACT410", "MSK-IMPACT468", "MSK-IMPACT505")
for (p in names(coveredMb)) {
  currPanel <- panelProbes[which(panelProbes$SEQ_ASSAY_ID == p), ]
  coveredMb[p] <- (sum(currPanel$End_Position - currPanel$Start_Position)) / 1000000
}

# chromosome band info for hg19, obtained from UCSC GB
chrBandInfo <- read.delim(paste0(baseDir, "data/misc/chrBandInfo.txt"))

# sv file from GENIE 14
sv <- read.delim(paste0(baseDir, "data/data_sv.txt"))

################################################################################




##### 02: SAMPLE SELECTION
################################################################################

# set a new name for the dataset to be produced and create dir
projName <- "G14_LUAD"
if (!dir.exists(paste0(outDir, projName, "/"))) {dir.create(paste0(outDir, projName, "/"))}

# filter MSK samples by cancer type LUAD
desiredSamples <- sampleData[which(grepl("-MSK-", sampleData$SAMPLE_ID, fixed = T)), ] 
desiredSamples <- desiredSamples[which(desiredSamples$ONCOTREE_CODE %in% c("LUAD")), ]

# convert to numeric - this produces true NAs 
desiredSamples$AGE_AT_SEQ_REPORT <- as.numeric(desiredSamples$AGE_AT_SEQ_REPORT)

# init output
selectionRecord <- data.frame(matrix(nrow = 0, ncol = 8))
colnames(selectionRecord) <- c("patientID", "primID", "metaID", "paired", "nPrim", "nMeta", "metaStatus", "surgeryToLastContact")

# iterate over selected patients
for (uPat in unique(desiredSamples$PATIENT_ID)) {
  
  # get samples
  allPat <- desiredSamples[which(desiredSamples$PATIENT_ID == uPat), ]
  allPrim <- allPat[which(allPat$SAMPLE_TYPE == "Primary"), ]
  allMeta <- allPat[which(allPat$SAMPLE_TYPE == "Metastasis"), ]
  
  # initialise new output row 
  nPrim <- 0; nMeta <- 0; primID <- NA; metaID <- NA; paired <- FALSE; metaStatus <- NA; surgeryToLastContact <- NA
  
  # carry out sample selection rationale as described
  if (nrow(allPrim) > 0) {
    
    # sort samples according to the following priorities
    # 1. avoid sampling time NAs
    # 2. prefer samples at lower sampling age
    # 3. prefer newer version of IMPACT assay
    allPrim <- allPrim[order(allPrim$SEQ_ASSAY_ID, decreasing = T, na.last = T), ]
    allPrim <- allPrim[order(allPrim$AGE_AT_SEQ_REPORT, decreasing = F, na.last = T), ]
    
    # choose best
    primID <- allPrim$SAMPLE_ID[1]; nPrim <- nrow(allPrim)
    
    if (nrow(allMeta) > 0) {
      
      # sort samples according to the following priorities
      # 1. avoid sampling time NAs
      # 2. prefer samples at higher sampling age
      # 3. prefer newer version of IMPACT assay
      allMeta <- allMeta[order(allMeta$SEQ_ASSAY_ID, decreasing = T, na.last = T), ]
      allMeta <- allMeta[order(allMeta$AGE_AT_SEQ_REPORT, decreasing = T, na.last = T), ]
      
      # choose best
      metaID <- allMeta$SAMPLE_ID[1]; nMeta <- nrow(allMeta); paired <- T
      metaStatus <- "isPaired"
      
    } else {
      
      # get info on metastatic status from unpaired primary (if available)
      mskID <- gsub("GENIE-MSK-", "", primID)
      if (mskID %in% metSampleDat$SAMPLE_ID) {
        
        if (metSampleDat[which(metSampleDat$SAMPLE_ID == mskID), "MET_COUNT"] == 0) {
          metaStatus <- "absent"
        } else if (metSampleDat[which(metSampleDat$SAMPLE_ID == mskID), "MET_COUNT"] >= 1) {
          metaStatus <- "present"
        }
        
        # record follow-up time. warnings from here are okay, NAs get replaced with -Inf which 
        mskPat <- metSampleDat[which(metSampleDat$SAMPLE_ID == mskID), "PATIENT_ID"]
        surgeryToLastContact <- max(metPatDat[which(metPatDat$PATIENT_ID == mskPat), "AGE_AT_DEATH"], metPatDat[which(metPatDat$PATIENT_ID == mskPat), "AGE_AT_LAST_CONTACT"], na.rm = T) - metPatDat[which(metPatDat$PATIENT_ID == mskPat), "AGE_AT_SURGERY"]
        
      } else {
        
        if (uPat %in% genieMetPatients) {
          metaStatus <- "present"
        } else {
          metaStatus <- "unknown"
        }
        
      }
      
      
      
    }
    
    
  }  else if (nrow(allMeta) > 0) {
    
    # sort samples according to the following priorities
    # 1. avoid sampling time NAs
    # 2. prefer samples at lower sampling age
    # 3. prefer newer version of IMPACT assay
    allMeta <- allMeta[order(allMeta$SEQ_ASSAY_ID, decreasing = T, na.last = T), ]
    allMeta <- allMeta[order(allMeta$AGE_AT_SEQ_REPORT, decreasing = T, na.last = T), ]
    
    
    metaID <- allMeta$SAMPLE_ID[1]; nMeta <- nrow(allMeta)
    metaStatus <- "isMetastasis"
  }
  
  # add to output
  npr <- data.frame(matrix(data = c(uPat, primID, metaID, paired, nPrim, nMeta, metaStatus, surgeryToLastContact), nrow = 1, ncol = ncol(selectionRecord)))
  colnames(npr) <- colnames(selectionRecord)
  
  selectionRecord <- rbind(selectionRecord, npr)
  
}

# convert to numeric
selectionRecord[, c("nPrim", "nMeta", "surgeryToLastContact")] <- lapply(selectionRecord[, c("nPrim", "nMeta", "surgeryToLastContact")], as.numeric)

## remove patients without valid sample (happens if samples are of unspecified type only)
selectionRecord <- selectionRecord[which(selectionRecord$nPrim + selectionRecord$nMeta > 0), ]

# get all included samples
desiredSamples <- sampleData[which(sampleData$SAMPLE_ID %in% union(selectionRecord$primID, selectionRecord$metaID)[which(!is.na(union(selectionRecord$primID, selectionRecord$metaID)))]), ] 

# add tmb calcs
selectionRecord[, c("primTMB_GENIE", "metaTMB_GENIE")] <- NA

for (ssr in 1:nrow(selectionRecord)) {
  
  if (is.na(selectionRecord[ssr, "primID"])) {
    # don't do anything
  } else {
    currSID <- selectionRecord[ssr, "primID"]
    currPanel <- sampleData[which(sampleData$SAMPLE_ID == currSID), "SEQ_ASSAY_ID"]
    selectionRecord[ssr, "primTMB_GENIE"] <- nrow(muts[which(muts$Tumor_Sample_Barcode == currSID), ]) / coveredMb[currPanel]
  }
  
  if (is.na(selectionRecord[ssr, "metaID"])) {
    # don't do anything
  } else {
    currSID <- selectionRecord[ssr, "metaID"]
    currPanel <- sampleData[which(sampleData$SAMPLE_ID == currSID), "SEQ_ASSAY_ID"]
    selectionRecord[ssr, "metaTMB_GENIE"] <- nrow(muts[which(muts$Tumor_Sample_Barcode == currSID), ]) / coveredMb[currPanel]
  }
}



##### 03: CNA EVENT DEFINITIONS AND CALLING
################################################################################

# subset seg file to included samples
seg <- seg[which(seg$ID %in% desiredSamples$SAMPLE_ID), ]
if (!grepl("chr", seg[1, "chromosome"])) {seg$chromosome <- paste0("chr", seg$chromosome)}

cat(paste0(nrow(desiredSamples), " samples were chosen", "\n",
           length(unique(seg$ID)), " (", round((length(unique(seg$ID))/nrow(desiredSamples))*100, 2), "%) of which have mecan-adjusted CN data available", "\n"))

# get the samples that do not have calibrated data - presumably CN silent
noCnDataSamples <- desiredSamples$SAMPLE_ID[which(desiredSamples$SAMPLE_ID %notin% seg$ID)]

# make minimum coordinate system

# get all unique positions by concatenating chromosome and every BP start/end
allPos <- sort(c(paste0(seg$chromosome, ":", seg$start), paste0(seg$chromosome, ":", seg$end)))
allUniquePos <- unique(allPos) # 8071 unique positions
posTab <- table(allPos)[order(table(allPos), decreasing = T)]

# check distribution of breakpoint incidence counts
plot(hist(log10(posTab)))

length(which(posTab == 1))
length(which(posTab < 3))

# initialise table format and fill
allUPTab <- data.frame(matrix(nrow = 0, ncol = 2))
for (i in allUniquePos) {
  posChr <- strsplit(i, ":", fixed = T)[[1]][1]
  posBas <- as.numeric(strsplit(i, ":", fixed = T)[[1]][2])
  nAUPT <- data.frame(matrix(nrow = 1, ncol = 2, data = c(posChr, posBas)))
  allUPTab <- rbind(allUPTab, nAUPT)
}
colnames(allUPTab) <- c("chr", "position")

# reformatting and ordering along the linear genome
allUPTab$position <- as.numeric(allUPTab$position)
allUPTab$chr <- factor(allUPTab$chr, levels=c(paste0("chr", c(1:22, "X"))))
allUPTab <- allUPTab[order(allUPTab$chr, allUPTab$position), ]
rownames(allUPTab) <- 1:nrow(allUPTab)

# adding unique "coordinate-IDs"
allUPTab$id <- NA
for (chr in unique(allUPTab$chr)) {
  allUPTab[which(allUPTab$chr == chr), "id"] <- paste0(chr, "-", 1:nrow(allUPTab[which(allUPTab$chr == chr), ]))
}

# convert to GR object
GRallUPTab <- GRanges(seqnames = allUPTab$chr, ranges = IRanges(allUPTab$position, allUPTab$position))

# add gene-level info: if probe position overlaps a COSMIC gene, add HGNC symbol to row
seqlevels(glGR) <- paste0("chr", seqlevels(glGR))
probeGeneMatches <- findOverlaps(GRallUPTab, glGR, type = "within")
allUPTab$inGene <- NA
for (pgm in 1:length(probeGeneMatches)) {
  currProbe <- probeGeneMatches@from[pgm]
  currGene <- probeGeneMatches@to[pgm]
  allUPTab[currProbe, "inGene"] <- geneLoci[currGene, "Gene.Symbol"]
}


# add "fraction of chromosome passed" to each coordinate, giving a sense of absolute
# position of the probes and allowing to see irregular probe distributions
allUPTab$chrFrac <- 0
for (uChr in unique(allUPTab$chr)) {
  currIndices <- which(allUPTab$chr == uChr)
  # get chromosome length
  totLen <- max(as.numeric(allUPTab[currIndices, "position"]))
  # calc fractions
  allUPTab[currIndices, "chrFrac"] <- round((as.numeric(allUPTab[currIndices, "position"]) / totLen), 4)
}



# get arm level coords
chrOrder <- paste0("chr", c(1:22, "X", "Y"))
chrBandInfo <- chrBandInfo[which(chrBandInfo$gieStain %notin% c("acen", "gvar", "stalk")), ]
chrBandInfo$X.chrom <- factor(chrBandInfo$X.chrom, levels = chrOrder)
chrBandInfo <- chrBandInfo[order(chrBandInfo$X.chrom), ]
chrBandInfo$arm <- paste0(gsub("chr", "", chrBandInfo$X.chrom), substr(chrBandInfo$name, 1, 1))

chrArms <- data.frame(matrix(nrow = 0, ncol = 4)); colnames(chrArms) <- c("chr", "arm", "start", "end")
for (arm in unique(chrBandInfo$arm)) {
  nRow <- data.frame(matrix(data=c(substr(arm, 1, nchar(arm)-1),
                                   arm,
                                   min(chrBandInfo[which(chrBandInfo$arm == arm), "chromStart"]),
                                   max(chrBandInfo[which(chrBandInfo$arm == arm), "chromEnd"])),
                            nrow = 1, ncol = 4))
  colnames(nRow) <- colnames(chrArms)
  chrArms <- rbind(chrArms, nRow)
}
chrArms <- chrArms[which(chrArms$chr != "Y"), ]

armsGR <- GRanges(seqnames = paste0("chr", chrArms$chr), ranges = IRanges(start = as.numeric(chrArms$start), end = as.numeric(chrArms$end)))
armsGR$arm <- chrArms$arm


# assign probes to arms
probeArmMatches <- findOverlaps(GRallUPTab, armsGR, type = "within")
allUPTab$arm <- NA
for (pam in 1:length(probeArmMatches)) {
  allUPTab[probeArmMatches[pam]@from, "arm"] <- armsGR[probeArmMatches[pam]@to]$arm
}

# assign probes to centromeres if they couldn't be assigned to arms
allUPTab[which(is.na(allUPTab$arm)), "arm"] <- "cen"






## fill the main matrices detailing per-probe per-sample event absence/presence

# initialise matrices, one for amps, one for dels
ampProbes <- data.frame(matrix(nrow = length(unique(seg$ID)), ncol = nrow(allUPTab), data = NA))
rownames(ampProbes) <- unique(seg$ID); colnames(ampProbes) <- allUPTab$id
delProbes <- ampProbes

# populate the matrix
pb <- txtProgressBar(min=0, max=length(unique(seg$ID)), initial = 0, style=3)
for (dS in unique(seg$ID)) {
  
  # for each sample..
  subSeg <- seg[which(seg$ID == dS), ]
  
  # subset its segments for amps/dels and convert remains to GR objects
  subAmp <- subSeg[which(subSeg$roundCN %in% c(3, "4+")), ]
  subDel <- subSeg[which(subSeg$roundCN %in% c(0, 1)), ]
  GRsubAmp <- GRanges(seqnames = subAmp$chromosome, ranges = IRanges(start = subAmp$start, end = subAmp$end))
  GRsubDel <- GRanges(seqnames = subDel$chromosome, ranges = IRanges(start = subDel$start, end = subDel$end))
  GRsubAmp$CN <- subAmp$roundCN; GRsubDel$CN <- subDel$roundCN
  
  # find probes overlapping with amps and add info to main table
  currAmpOverlaps <- findOverlaps(GRallUPTab, GRsubAmp, type = "within")
  currAmpProbes <- currAmpOverlaps@from
  currAmpVector <- rep(2, nrow(allUPTab))
  currAmpVector[currAmpProbes] <- rep(GRsubAmp$CN[rle(currAmpOverlaps@to)$values], times = rle(currAmpOverlaps@to)$lengths)
  ampProbes[dS, ] <- currAmpVector
  
  # find probes overlapping with dels and add info to main table
  currDelOverlaps <- findOverlaps(GRallUPTab, GRsubDel, type = "within")
  currDelProbes <- currDelOverlaps@from
  currDelVector <- rep(2, nrow(allUPTab))
  currDelVector[currDelProbes] <- rep(GRsubDel$CN[rle(currDelOverlaps@to)$values], times = rle(currDelOverlaps@to)$lengths)
  delProbes[dS, ] <- currDelVector
  
  # update progress bar
  if (which(unique(seg$ID)==dS) %% 20 == 0) {
    setTxtProgressBar(pb, which(unique(seg$ID)==dS))
  }
  
}

close(pb)



# add as stacked/collapsed info to probe table
ampBinary <- ampProbes; ampBinary[ampBinary != 2] <- 1; ampBinary[ampBinary == 2] <- 0
delBinary <- delProbes; delBinary[delBinary != 2] <- 1; delBinary[delBinary == 2] <- 0
allUPTab$ampSum <- NA; allUPTab$delSum <- NA
for (probe in 1:nrow(allUPTab)) {
  allUPTab[probe, "ampSum"] <- sum(as.numeric(ampBinary[, allUPTab[probe, "id"]]))
  allUPTab[probe, "delSum"] <- sum(as.numeric(delBinary[, allUPTab[probe, "id"]]))
}

# save alteration frequencies per coordinate
if (!dir.exists(paste0(outDir, projName, "/CNA/"))) {dir.create(paste0(outDir, projName, "/CNA/"))}
write.table(allUPTab, paste0(outDir, projName, "/CNA/coordinateDetails.txt"), quote = F, sep = "\t")

# manually pick CN events; 1 per chr; based on top frequency or known driver

ampEvents <- data.frame(start = c("chr1-453", "chr5-31", "chr7-104", "chr12-113"),
                        end = c("chr1-456", "chr5-46", "chr7-132", "chr12-117"),
                        target = c("MCL1/1q", "TERT/5p", "EGFR/7p", "KRAS/12p"))


delEvents <- data.frame(start = c("chr9-112", "chr13-147", "chr17-19", "chr19-5"),
                        end = c("chr9-113", "chr13-149", "chr17-20", "chr19-6"),
                        target = c("CDKN2A/9p", "RB1/13q", "TP53/17p", "STK11/19p"))

# make "figure-ready" names
ampEvents$ID <- paste0(ampEvents$target, " (Amp)")
delEvents$ID <- paste0(delEvents$target, " (Del)")


## check presumed CN silent samples

# get original CN data of samples without calibrated output
silentCNSamplesMSK <- originalSegMSK[which(originalSegMSK$ID %in% noCnDataSamples), ]

# check their LRR distributions
summary(silentCNSamplesMSK$seg.mean)
hist(silentCNSamplesMSK$seg.mean, breaks = 50) # indeed they look very silent
abline(v=0.15)
abline(v=-0.15)

# add empty rows to amp/del probe tables for checked silent CN samples
silentCN <- data.frame(matrix(data = 0, nrow = length(noCnDataSamples), ncol = ncol(ampBinary)))
rownames(silentCN) <- noCnDataSamples; colnames(silentCN) <- colnames(ampBinary)
ampBinary <- rbind(ampBinary, silentCN); delBinary <- rbind(delBinary, silentCN)

# sort like input clinical data file
ampBinary <- ampBinary[desiredSamples$SAMPLE_ID, ]; delBinary <- delBinary[desiredSamples$SAMPLE_ID, ]
ampBinary[, 1:ncol(ampBinary)] <- sapply(ampBinary, as.numeric)
delBinary[, 1:ncol(delBinary)] <- sapply(delBinary, as.numeric)

## save data
write.table(ampBinary, paste0(outDir, projName, "/CNA/ampProbes.txt"), quote = F, sep = "\t")
write.table(delBinary, paste0(outDir, projName, "/CNA/delProbes.txt"), quote = F, sep = "\t")



## call events in samples by checking whether defined intervals are completely amplified/deleted

# init output
cnEventRecord <- data.frame(matrix(nrow = nrow(desiredSamples), ncol = 0))
rownames(cnEventRecord) <- rownames(ampBinary)

# call amps
for (ampR in 1:nrow(ampEvents)) {
  
  currIndices <- which(colnames(ampBinary) == ampEvents[ampR, "start"]) : which(colnames(ampBinary) == ampEvents[ampR, "end"])
  currProbes <- ampBinary[, currIndices, drop = F]
  cnEventRecord[, ampEvents[ampR, "ID"]] <- rowSums(currProbes) == ncol(currProbes)

}

# call dels
for (delR in 1:nrow(delEvents)) {
  
  currIndices <- which(colnames(delBinary) == delEvents[delR, "start"]) : which(colnames(delBinary) == delEvents[delR, "end"])
  currProbes <- delBinary[, currIndices, drop = F]
  cnEventRecord[, delEvents[delR, "ID"]] <- rowSums(currProbes) == ncol(currProbes)
  
}

## convert to binary
cnEventRecord <- cnEventRecord*1

## sort by frequency
cnEventRecord <- cnEventRecord[ , order(colSums(cnEventRecord), decreasing = T)]

# exchange probe coords for actual coords
for (a in 1:nrow(ampEvents)) {
  ampEvents[a, "chr"] <- allUPTab[which(allUPTab$id == ampEvents[a, "start"]), "chr"]
  ampEvents[a, "start"] <- allUPTab[which(allUPTab$id == ampEvents[a, "start"]), "position"]
  ampEvents[a, "end"] <- allUPTab[which(allUPTab$id == ampEvents[a, "end"]), "position"]
}
ampEvents <- ampEvents[, c(5, 1, 2, 3, 4)]

for (a in 1:nrow(delEvents)) {
  delEvents[a, "chr"] <- allUPTab[which(allUPTab$id == delEvents[a, "start"]), "chr"]
  delEvents[a, "start"] <- allUPTab[which(allUPTab$id == delEvents[a, "start"]), "position"]
  delEvents[a, "end"] <- allUPTab[which(allUPTab$id == delEvents[a, "end"]), "position"]
}
delEvents <- delEvents[, c(5, 1, 2, 3, 4)]

# save event definitions
write.table(ampEvents, paste0(outDir, projName, "/CNA/ampEvents.txt"), quote = F, sep = "\t")
write.table(delEvents, paste0(outDir, projName, "/CNA/delEvents.txt"), quote = F, sep = "\t")

################################################################################



##### 04: MUTATION EVENT CALLING
################################################################################

# first, check mutation validity: is gene measured consistently across arrays
geniePanels <- list.files("data/gene_panels/")

# get all solid tumor MSK panels
mskPanels <- geniePanels[which(grepl("MSK", geniePanels))]
mskPanels <- mskPanels[which(!grepl("ACCESS", mskPanels))]
mskPanels <- mskPanels[which(!grepl("HEME", mskPanels))]

# get their genes
genes341 <- colnames(read.delim(paste0("data/gene_panels/", mskPanels[1]), skip = 2))[-1]
genes410 <- colnames(read.delim(paste0("data/gene_panels/", mskPanels[2]), skip = 2))[-1]
genes468 <- colnames(read.delim(paste0("data/gene_panels/", mskPanels[3]), skip = 2))[-1]
genes505 <- colnames(read.delim(paste0("data/gene_panels/", mskPanels[4]), skip = 2))[-1]

# get intersect: genes assayed in all panels
validGenes <- intersect(intersect(genes341, genes410), intersect(genes468, genes505))

# apply functionality filter and subset for selected samples
mutsInSamples <- muts[which(muts$eventFilter), ]
mutsInSamples <- mutsInSamples[which(mutsInSamples$Tumor_Sample_Barcode %in% desiredSamples$SAMPLE_ID), ]

# init output
mutEvents <- data.frame(matrix(nrow = length(desiredSamples$SAMPLE_ID), ncol = 0))
rownames(mutEvents) <- desiredSamples$SAMPLE_ID

# per gene present in the samples, create a name, check the samples that have
# a mutation and add to output
mutEvNames <- c()
for (uMut in validGenes) {
  
  newName <- paste0(uMut, " (M)"); mutEvNames <- c(mutEvNames, newName)
  currMuts <- mutsInSamples[which(mutsInSamples$Hugo_Symbol == uMut), ]
  currBinary <- vector(mode = "numeric", length = length(desiredSamples$SAMPLE_ID))
  names(currBinary) <- desiredSamples$SAMPLE_ID
  currBinary[unique(currMuts$Tumor_Sample_Barcode)] <- 1
  
  mutEvents <- cbind(mutEvents, currBinary)
  
}

colnames(mutEvents) <- mutEvNames


# sort rows same way as CN events
mutEvents <- mutEvents[rownames(cnEventRecord), ]

# sort cols by freq
mutEvents <- mutEvents[, order(colSums(mutEvents), decreasing = T)]

plot(colSums(mutEvents) / nrow(mutEvents))

## sanity check
all(rownames(cnEventRecord) == rownames(mutEvents))

################################################################################




##### 06: EVENT SELECTION AND CONVERSION TO P/M FORMAT
################################################################################

# manually include these CNA events:
manualCnEvents <- colnames(cnEventRecord)

# get their indices 
manualCnEventsIndices <- which(colnames(cnEventRecord) %in% manualCnEvents)

# set how many muts are chosen based on frequency alone
mcMuts <- 20

# select those events plus the manually selected ones
select <- cbind(cnEventRecord[, c(0, manualCnEventsIndices)], mutEvents[, (0:mcMuts)])
select <- select[, order(colSums(select), decreasing = T)]
table(rowSums(select))

## convert into PM format: instead of observations = samples, do 
## observations = patients, with "P.[base event name]" columns detailing events
## present in the primary tumour and "M.[base event name]" columns detailing 
## events in the metastasis (automatically 0 if only PT/MT is available per 
## patient)

# make split event names
pmEvents <- c()
for (se in colnames(select)) {
  pmEvents <- c(pmEvents, paste0(c("P.", "M."), se))
}

# add binary if patient has paired samples
pmEvents <- c(pmEvents, "paired")

# init output
pmForm <- data.frame(matrix(ncol = length(pmEvents), nrow = 0))
colnames(pmForm) <- pmEvents

# for each patient in selection record
for (uPat in 1:nrow(selectionRecord)) {
  
  ssRow <- selectionRecord[uPat, ]
  
  newPMrow <- data.frame(matrix(ncol = length(pmEvents), nrow = 1, data = 0))
  colnames(newPMrow) <- pmEvents; rownames(newPMrow) <- ssRow$patientID
  
  # if it has primary sample included ..
  if (!is.na(ssRow$primID)) {
    
    # get the events active in the primary and set them to 1 in the about-to-be-added row
    pEvs <- paste0("P.", names(unlist(select[ssRow$primID, , drop = T])[which(unlist(select[ssRow$primID, , drop = T]) == 1)]))
    newPMrow[, pEvs[which(pEvs != "P.")]] <- 1
    
    # if it also has metastasis sample included ..
    if (!is.na(ssRow$metaID)) {
      
      # get the events active in the metastasis and set them to 1 in the about-to-be-added row
      mEvs <- paste0("M.", names(unlist(select[ssRow$metaID, , drop = T])[which(unlist(select[ssRow$metaID, , drop = T]) == 1)]))
      newPMrow[, mEvs[which(mEvs != "M.")]] <- 1
      # also, set paired flag to 1
      newPMrow[, "paired"] <- 1
      
    } 
  
    # if it has metastasis sample included instead ..  
  } else if (!is.na(ssRow$metaID)) {
    
    # get the events active in the metastasis and set them to 1 in the about-to-be-added row
    mEvs <- paste0("M.", names(unlist(select[ssRow$metaID, , drop = T])[which(unlist(select[ssRow$metaID, , drop = T]) == 1)]))
    newPMrow[, mEvs[which(mEvs != "M.")]] <- 1
    
  }
  
  # add new row to output
  pmForm <- rbind(pmForm, newPMrow)
  
}

# HERE IT NEEDS TO BE MANUALLY EVALUATED HOW MANY OBSERVATONS EXCEED EVENT LIMIT

# active event distribution with paired event
table(rowSums(pmForm)) 
# active event distribution without paired event
table(rowSums(pmForm[, colnames(pmForm)[which(colnames(pmForm) != "paired")]]))

# check observations with most active events
View(pmForm[order(rowSums(pmForm), decreasing = T), ])

# check paired observations only
pmFormPairedOnly <- pmForm[which(pmForm$paired == 1), ]
table(rowSums(pmFormPairedOnly))

# identify observations exceeding limit
omitHyperSamples <- rownames(pmForm[which(rowSums(pmForm[, colnames(pmForm)[which(colnames(pmForm) != "paired")]]) >= 24), ])

# implement sample omission
pmFormPairedOnlyOmitted <- pmFormPairedOnly[which(rownames(pmFormPairedOnly) %notin% omitHyperSamples), ]
pmFormOmitted <- pmForm[which(rownames(pmForm) %notin% omitHyperSamples), ]

# confirm all is within limit now
table(rowSums(pmFormOmitted))
table(rowSums(pmFormOmitted[, colnames(pmFormOmitted)[which(colnames(pmFormOmitted) != "paired")]]))


# update sample selection accordingly
selectionRecordOmitted <- selectionRecord[which(selectionRecord$patientID %notin% omitHyperSamples), ]

## double check
colSums(pmFormOmitted) / nrow(pmFormOmitted)
table(rowSums(pmFormOmitted))

## check dist of met age - prim age and add them to the table
sampleData <- read.delim(paste0(getwd(), "/data/data_clinical_sample.txt"), comment.char = "#")
sampleData$AGE_AT_SEQ_REPORT <- as.numeric(sampleData$AGE_AT_SEQ_REPORT)

pairedSampleSelection <- selectionRecord[which(as.logical(selectionRecord$paired)), ]

metAgeMinusPrimAge <- c()

for (pss in 1:nrow(pairedSampleSelection)) {
  primAge <- sampleData[which(sampleData$SAMPLE_ID == pairedSampleSelection[pss, "primID"]), "AGE_AT_SEQ_REPORT"]
  metAge <- sampleData[which(sampleData$SAMPLE_ID == pairedSampleSelection[pss, "metaID"]), "AGE_AT_SEQ_REPORT"]
  metAgeMinusPrimAge <- c(metAgeMinusPrimAge, (metAge - primAge))
}

plot(hist(metAgeMinusPrimAge, breaks = 50))

## add to table

pmFormOmitted$P.AgeAtSeqRep <- ""
pmFormOmitted$M.AgeAtSeqRep <- ""


for (ssRow in 1:nrow(selectionRecord)) {
  
  if (!is.na(selectionRecord[ssRow, "primID"])) {
    pmFormOmitted[selectionRecord[ssRow, "patientID"], "P.AgeAtSeqRep"] <- sampleData[which(sampleData$SAMPLE_ID == selectionRecord[ssRow, "primID"]), "AGE_AT_SEQ_REPORT"]
  } else {
    pmFormOmitted[selectionRecord[ssRow, "patientID"], "P.AgeAtSeqRep"] <- "No primary included"
  }
  
  if (!is.na(selectionRecord[ssRow, "metaID"])) {
    pmFormOmitted[selectionRecord[ssRow, "patientID"], "M.AgeAtSeqRep"] <- sampleData[which(sampleData$SAMPLE_ID == selectionRecord[ssRow, "metaID"]), "AGE_AT_SEQ_REPORT"]
  } else {
    pmFormOmitted[selectionRecord[ssRow, "patientID"], "M.AgeAtSeqRep"] <- "No metastasis included"
  }
  
}


# save outputs
write.csv(pmFormOmitted, paste0(getwd(), "/outputs/eventProjects/", projName, "/", projName, "_Events.csv"), row.names = T, quote = F)
write.csv(selectionRecordOmitted, paste0(getwd(), "/outputs/eventProjects/", projName, "/", projName, "_sampleSelection.csv"), row.names = F, quote = F)


################################################################################












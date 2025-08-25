MODULE NAME
dataframeanalysis

VERSIONS HISTORY
v1 Created on 14/7/17 by Francesco Mariottini (fm)
v2 Merging two modules and improve features on 16/8/17 by fm
v3 Removing tools related only to the solar environment on 6/9/18 by fm 

The module dataframe_quality was created by merging previous modules (dataframe_cleaner & dataframe_explorer) 


DESCRIPTION
This module provides basic functions and procedures to check (and eventually improve) data quality in dataframes 

TABLE OF CONTENT
The original structure was modified to keep tools which could be used in different sectors:
- Preview: basic functions to have a first look at a dataframe
- Completeness: function related to data completeness (e.g. missing days in records) 
- StatisticalMethods: statistical methods to assess data integrity (e.g. identification of outliers based on linear regression)  
- UserDefinedIntegrity: flag data integrity according to a defined checks table

Note: small description of external function/class may be retrieved through "__doc__" attribute.

API & RELATED MODULES
Check the dedicated readme file "README_API_&_MODULES.txt"

SUPPORT MODULES
pandas

CONTRIBUTING

CREDITS

LICENSE
Contact GitHub API Training Shop Blog About © 2017 GitHub, Inc. Terms Privacy Security Status Help

NOTE
README file could be later formatted according to the Markdown syntax (https://guides.github.com/features/mastering-markdown/) 
current documentation structure based on https://guides.github.com/features/wikis/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Downloader
===========

Helper file for fetching data from the PostgreSQL to a local CSV

Example:
        $ python file.py

"""

__author__ = "José Ramón Morera Campos"
__version__ = "1.0.1"
#######################################################################
import psycopg2
import csv

def export_table_to_csv(table_name: str):
    # Connection string (update credentials as needed)
    conn_str = "postgres://postgres:19@localhost/postgres"
    
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        
        # Create and execute the SQL query
        query = f"SELECT * FROM {table_name};"
        cur.execute(query)
        
        # Fetch all rows and column names
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        
        # Define the output CSV file name based on the table name
        output_file = f"{table_name}.csv"
        
        # Write data to the CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(col_names)  # write header
            writer.writerows(rows)      # write data rows
        
        print(f"Data from table '{table_name}' has been exported to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up database resources
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

######################################
# export_table_to_csv("openmeteo_cristianos_arpege")
# export_table_to_csv("openmeteo_cristianos_icon")
# export_table_to_csv("openmeteo_cuesta_arpege")
# export_table_to_csv("openmeteo_cuesta_icon")
# export_table_to_csv("openmeteo_orotava_arpege")
# export_table_to_csv("openmeteo_orotava_icon")
# export_table_to_csv("openmeteo_arona_arpege")
# export_table_to_csv("openmeteo_arona_icon")
# export_table_to_csv("openmeteo_santa_cruz_arpege")
# export_table_to_csv("openmeteo_santa_cruz_icon")
# export_table_to_csv("openmeteo_santiago_teide_arpege")
# export_table_to_csv("openmeteo_santiago_teide_icon")
export_table_to_csv("openmeteo_garachico_arpege")
export_table_to_csv("openmeteo_garachico_icon")
# export_table_to_csv("openmeteo_adeje_icon")
# export_table_to_csv("openmeteo_adeje_arpege")
# export_table_to_csv("openmeteo_el_rosario_icon")
# export_table_to_csv("openmeteo_el_rosario_arpege")
# export_table_to_csv("openmeteo_punta_hidalgo_icon")
# export_table_to_csv("openmeteo_punta_hidalgo_arpege")

# export_table_to_csv("openmeteo_buenavista_norte_icon")
# export_table_to_csv("openmeteo_buenavista_norte_arpege")

#export_table_to_csv("grafcan_santa_cruz")
# export_table_to_csv("grafcan_cristianos")
# export_table_to_csv("grafcan_cuesta")
# export_table_to_csv("grafcan_orotava")
# export_table_to_csv("grafcan_arona")
# export_table_to_csv("grafcan_santiago_teide")
# export_table_to_csv("grafcan_garachico")
# export_table_to_csv("grafcan_buenavista_norte")
# export_table_to_csv("grafcan_adeje")
# export_table_to_csv("grafcan_el_rosario")
# export_table_to_csv("grafcan_punta_hidalgo")

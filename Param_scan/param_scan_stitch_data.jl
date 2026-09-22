using JLD2, DataFrames

# Include 2 to loop over both JJ and MultiSite arrays
sites_list = 2:20 
regimes = ["s", "l"]  # small and large thermalisation regimes
params = ["J", "gamma", "n_avg", "M", "B_21", "Delta", "T"]

println("Stitching cluster JLD2 files together...")

for sites in sites_list
    for regime in regimes
        all_dfs = DataFrame[]
        processed_files = String[] 
        
        # Check if any files exist for this site/regime combination
        files_found_count = 0
        
        for p in params
            file = "Param_scan/Data/Results_$(sites)_$(regime)_$(p).jld2"
            if isfile(file)
                # Load the DataFrame stored inside the JLD2 file
                data = load(file)
                df = data["df"]
                
                push!(all_dfs, df)
                push!(processed_files, file)
                files_found_count += 1
            end
        end
        
        # Logic: Only stitch if we found ALL parameters for this set
        # This prevents creating incomplete master files if a job is still running
        if files_found_count == length(params)
            # 1. Stitch and Save Master File
            master_df = vcat(all_dfs...)
            out_file = "Param_scan/Data/Master_Results_$(sites)_Site_$(regime).jld2"
            
            # Save the stitched dataframe back to JLD2
            jldsave(out_file; df = master_df)
            println("Successfully created: $out_file")

            # 2. Cleanup: Delete the small temporary files
            for f in processed_files
                rm(f)
            end
            println("   -> Cleaned up $(length(processed_files)) temporary files for $(sites) sites.")
            
        elseif files_found_count > 0
            # Some files found but not all? Notify the user but don't delete anything yet
            println("SKIPPING Sites: $sites ($regime): Only $files_found_count/$(length(params)) files ready.")
        end
    end
end

println("\nAll available data stitched and scratch space cleaned!")
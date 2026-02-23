Insurance App for agricultural fields using Satellite Data Climate-related disasters—such as floods, extreme temperatures, and droughts—are becoming more frequent and unpredictable. Insurance companies struggle to accurately assess land risk, often relying on costly, manual field inspections. This inefficiency leads to high operational costs, inaccurate risk assessments, and increased financial exposure.

Our Solution: We provide an ML-powered platform leveraging satellite data to help large insurance firms assess land risk in real-time. By using machine learning and geospatial analysis, we eliminate the need for on-site specialists, offering faster, more precise risk evaluations while reducing costs. By gathering high-resolution satellite imagery and climate data, our AI models analyze historical and real-time conditions to detect flood risk, drought potential, and extreme weather trends. Insurers are then able to receive instant, automated risk reports—integrated seamlessly into their underwriting process.

How to run: 
0. Download the river dataset from the link: https://drive.google.com/drive/folders/1BN7roJyW6wZ-ibY5VkX-HSB80EFvyfJJ?usp=sharing and place them at Stack-Overflood/used_data/rivers_final/
1. In case it is needed, change path to the geopackage files accordingly inside the CSV: euhydro_tile_index_25km.csv
2. In case it is needed, change results path in BatchProcessParallel.py
3. Add interest locations and date in Stack-Overflood/used_data/entries.csv, under the header, in center_lat,center_lon,center_date format
4. Set the desired OFFSET inside the BatchProcessParallel.py
5. Run BatchProcessParallel.py

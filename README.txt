Field Library Instrument README

An instrument for environmental sound analysis, informed by the identification of acoustic similarities within selected sound libraries (using a matching pursuit approach).

The instrument is based on two components: a narrative track (N track) and a voice track (V track). The N track provides the initial structure, onto which the model introduces audio clips from the V track library that share similar acoustic properties.

Code structure
        main.py
            Original exploration into .py

        list_files.py
            Displays audio tracks in folder

        nv_setup.py
            Selection of N tracks and V tracks
        
        nv_load.py
            Selection of N tracks and V track and identification of properties (name, sampler rate, duration, max amplitude,etc.)

        nv_compare.py   
            Presents comparison results of V tracks againt N track

        nv_matching_engine.py
            Based on compartive results, analyses best whole V track to selected N track

        nv_adaptive_match.py
            Analyses nuances in the V tracks and compiles possible matches

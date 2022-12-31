cd build/
pprof --text test_lk_greftools test.prof
pprof --pdf test_lk_greftools test.prof >test.pdf
cd ..
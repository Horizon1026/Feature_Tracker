cd src/dense_optical_flow_tracker/
python3 raft.py
cd ../..

cd build/
./test_optical_flow
./test_direct_method
./test_descriptor_matcher_brief
./test_descriptor_matcher_xfeat
cd ..

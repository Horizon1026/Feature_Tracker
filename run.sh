cd src/dense_optical_flow_tracker/
python3 raft.py
cd ../..

cd build/
./test_optical_flow
./test_direct_method
./test_descriptor_matcher_brief
./test_descriptor_matcher_superpoint
./test_descriptor_matcher_disk
./test_nn_feature_matcher
cd ..

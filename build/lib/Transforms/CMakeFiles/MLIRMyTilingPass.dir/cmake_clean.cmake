file(REMOVE_RECURSE
  "../libMLIRMyTilingPass.a"
  "../libMLIRMyTilingPass.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRMyTilingPass.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

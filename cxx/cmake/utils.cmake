

# ex: list_contains(CUDA_ARCHS "sm_20" result)
function(list_contains list_var element result_var)
    set(found FALSE)
    foreach(item ${${list_var}})
        if(${item} STREQUAL ${element})
            set(found TRUE)
            break()
        endif()
    endforeach()
    set(${result_var} ${found} PARENT_SCOPE)
endfunction()

# list1 - list2
function(list_diff list1 list2 result_var)
    set(temp1 ${list1})
    set(temp2 ${list2})
    set(result_tmp)

    foreach(elem1 ${list1})
        list(FIND temp2 ${elem1} index) 
        if(${index} EQUAL -1)
            list(APPEND result_tmp ${elem1})
        endif()
    endforeach()
    
    set(${result_var} ${result_tmp} PARENT_SCOPE)
    
endfunction()

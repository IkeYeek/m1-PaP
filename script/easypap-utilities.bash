#!/usr/bin/env bash

_script_dir=$(dirname $BASH_SOURCE)
EASYPAPDIR=${EASYPAPDIR:-$(realpath ${_script_dir}/..)}
. ${_script_dir}/easypap-common.bash
unset _script_dir

# result placed in kernels
_easypap_kernels()
{
    local f tmp

    kernels=

    if [ ! -f obj/none.o ]; then
        return
    fi

    tmp=`nm obj/*.o | awk '/_compute_seq$/ {print $3}'`

    for f in $tmp; do
        f=${f#_}
        f=${f%_compute_seq}
        kernels="$kernels $f"
    done
}

# result placed in variants
_easypap_variants()
{
    local f p file tmp obj v

    variants=
    obj=

    if (( $# == 0 )); then
        set none
    fi

    for p in "${KERNEL_PREFIX[@]}" ; do
        file=obj/$p$1.o
        if [ -f $file ]; then
            obj="$obj $file"
        fi
    done

    if [[ -z $obj ]]; then
        return
    fi

    tmp=`nm $obj | awk '/ +_?'"$1"'_compute_[^.]*$/ {print $3}'`

    for f in $tmp; do
        v=${f#*_compute_}
        if [[ $v =~ ^ocl* || $v =~ ^cuda* ]]; then
            continue
        fi
        variants="$variants $v"
    done
}

# result places in gpu_flavor
_easypap_gpu_flavor()
{
    gpu_flavor=$(./run --gpu-flavor)
}

# result placed in gvariants
_easypap_opencl_variants()
{
    local f tmp k

    # The most secure way of finding GPU kernels is to ask easypap…
    tmp=$(./run -k $1 -lgv 2> /dev/null)

    # But the "grep into the .cl file" method is much much faster!
    #tmp=`awk '/__kernel/ {print $3}' < kernel/ocl/${k}.cl`

    for f in $tmp; do
        if [[ $f =~ ^$1_ocl* ]]; then
            gvariants="$gvariants ${f#$1_}"
        fi
    done
}

# result placed in gvariants
_easypap_cuda_variants()
{
    local f p file tmp obj v

    obj=

    file=obj/cuda_$1.o
    if [ -f $file ]; then
        obj="$obj $file"
    fi

    if [[ -z $obj ]]; then
        return
    fi

    tmp=`nm $obj | awk '/ +_?'"$1"'_cuda[^.]*$/ {print $3}'`

    for f in $tmp; do
        if [[ $f =~ ^$1_cuda* ]]; then
            gvariants="$gvariants ${f#$1_}"
        fi
    done
}

# result placed in gvariants
_easypap_gpu_variants()
{
    gvariants=

    if (( $# == 0 )); then
        set none
    fi

    _easypap_gpu_flavor
    case $gpu_flavor in
        ocl)
            _easypap_opencl_variants $1
            ;;
        cuda)
            _easypap_cuda_variants $1
            ;;
        *)
            ;;
    esac
}

# result placed in draw_funcs
_easypap_draw_funcs()
{
    local f p file tmp obj

    draw_funcs=
    obj=

    if (( $# == 0 )); then
        set none
    fi

    for p in "${KERNEL_PREFIX[@]}" ; do
        file=obj/$p$1.o
        if [ -f $file ]; then
            obj="$obj $file"
        fi
    done

    if [[ -z $obj ]]; then
        return
    fi

    tmp=`nm $obj | awk '/ +_?'"$1"'_draw_[^.]*$/ {print $3}'`

    for f in $tmp; do
        draw_funcs="$draw_funcs ${f#*_draw_}"
    done
}

# result placed in tile_funcs
_easypap_tile_funcs()
{
    local f p file tmp obj

    tile_funcs=
    obj=

    if (( $# == 0 )); then
        set none
    fi

    for p in "${KERNEL_PREFIX[@]}" ; do
        file=obj/$p$1.o
        if [ -f $file ]; then
            obj="$obj $file"
        fi
    done

    if [[ -z $obj ]]; then
        return
    fi

    tmp=`nm $obj | awk '/ +_?'"$1"'_do_tile_[^.]*$/ {print $3}'`
    if [[ -n $tmp ]]; then
        for f in $tmp; do
            tile_funcs="$tile_funcs ${f#*_do_tile_}"
        done
        return
    fi
    tmp=`nm $obj | awk '/ +_?'"$1"'_do_patch_[^.]*$/ {print $3}'`
    for f in $tmp; do
        tile_funcs="$tile_funcs ${f#*_do_patch_}"
    done
}

# result placed in pos
_easypap_option_position()
{
    local p

    if [[ $1 =~ ^--.* ]]; then
	    list=("${LONG_OPTIONS[@]}")
    else
	    list=("${SHORT_OPTIONS[@]}")
    fi
    for (( p=0; p < $NB_OPTIONS; p++ )); do
	    if [[ "${list[p]}" = "$1" ]]; then
	        pos=$p
	        return
	    fi
    done
    pos=$NB_OPTIONS
}

_easypap_remove_from_suggested()
{
    local i
    for (( i=0; i < ${#suggested[@]}; i++ )); do 
        if [[ ${suggested[i]} == $1 ]]; then
            suggested=( "${suggested[@]:0:$i}" "${suggested[@]:$((i + 1))}" )
            i=$((i - 1))
        fi
    done
}

_easypap_option_suggest()
{
    local c e o suggested=("$@")

    # We should prune the options than should only appear at first position    
    for (( o=0; o < $NB_OPTIONS; o++ )); do
        local short=${SHORT_OPTIONS[o]}
        local only_in_first_place_opt=only_in_first_place_${short#-}
        eval only_in_first_place_opt="\$${only_in_first_place_opt}"
        if (( only_in_first_place_opt == 1 )); then
            _easypap_remove_from_suggested ${LONG_OPTIONS[o]}
            _easypap_remove_from_suggested $short
           fi
    done

    for (( c=1; c < $COMP_CWORD; c++ )); do
	    if [[ ${COMP_WORDS[c]} =~ ^-.* ]]; then
	        _easypap_option_position ${COMP_WORDS[c]}
	        if (( pos < NB_OPTIONS )); then
	            local short=${SHORT_OPTIONS[pos]}
                local multiple_opt=multiple_${short#-}
                eval multiple_opt="\$${multiple_opt}"

                if [[ -z $multiple_opt ]]; then
		            # we shall remove this option from suggested options
                    _easypap_remove_from_suggested ${LONG_OPTIONS[pos]}
                    _easypap_remove_from_suggested $short
                fi

		        # also remove antagonist options
		        local excluding_opt=exclude_${short#-}
		        eval excluding_opt='(${'${excluding_opt}'[@]})'
		        for ((e=0; e < ${#excluding_opt[@]}; e++ )); do
		            local p=${excluding_opt[e]}
                    _easypap_remove_from_suggested ${LONG_OPTIONS[p]}
                    _easypap_remove_from_suggested ${SHORT_OPTIONS[p]}
		        done
	        fi
	    fi
    done

    COMPREPLY=($(compgen -W '"${suggested[@]}"' -- $cur))
}

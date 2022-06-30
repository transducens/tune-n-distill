#! /bin/bash
set -euo pipefail

LANG_1=$1
LANG_2=$2

lang_1_copy=(${LANG_1//_/ })
sub_lang_1=${lang_1_copy[0]} #normal lang code

lang_2_copy=(${LANG_2//_/ })
sub_lang_2=${lang_2_copy[0]} #normal lang code

# Experiments path
EXPERIMENT_DIR=$3

# Corpora path. train, dev, test, mono
CORPORA_DIR=$3/corpora

#varible from launch-train.sh
MBART50_DIR=$4
MBART50_N1=$MBART50_DIR/mbart50-n1
MBART50_1N=$MBART50_DIR/mbart50-1n

TOOLS=$PWD/tools

BEST_BLEU_1N="" # best 1-n BLEU
BEST_BLEU_N1="" # best n-1 BLEU

#Paths of best models for each direction
BEST_N1=$MBART50_N1
BEST_1N=$MBART50_1N

BICLEANER_SCORE_N1="" #${score:0:1}.${score:1:2}
BICLEANER_SCORE_1N=""

mono_1_original=""
mono_n_synthetic=""

mono_n_original=""
mono_1_synthetic=""


function prepare_dir() {
        local IT=$1
        local MODEL=$2
        local SOURCE=$3
        local TARGET=$4

        local MONO_SYNTHETIC=$5
        local MONO_ORIGINAL=$6
	
	if [[ ! -f $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE ]]; then

        	mkdir $EXPERIMENT_DIR/it-$IT-$MODEL # ej: it_2_mbart50-n1
        	mkdir $EXPERIMENT_DIR/it-$IT-$MODEL/corpora

        	for lang in $SOURCE $TARGET; do
        	        cp $CORPORA_DIR/train.$lang $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$lang
        	done

		if ! [[ $MONO_SYNTHETIC = null ]] && ! [[ $MONO_ORIGINAL = null ]]; then
        		cat $MONO_SYNTHETIC >> $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE # Add synthttic corpus to source
        		cat $MONO_ORIGINAL >> $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$TARGET # Add original corpus to target
				paste $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$TARGET > $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET
                awk -F "\t" '$1!="" && $2!="" {print $1, "\t", $2}' $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET > $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET.clean
                cut -f1 $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET.clean > $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE
                cut -f2 $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET.clean > $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$TARGET
                rm $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET
                rm $EXPERIMENT_DIR/it-$IT-$MODEL/corpora/train.$SOURCE-$TARGET.clean
		fi
	else
		echo "Directory alredy exist" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi

}

function spm_encode() {
	local l1=$1
	local l2=$2
	local spm_model_dir=$3
	local corpora=$4

	if [[ ! -f $corpora/train.spm.$l1 ]]; then
		for lang in $l1 $l2
			do 
				python $TOOLS/spm_encode.py --model $spm_model_dir/spm.model --output_format=piece --inputs=$corpora/train.$lang --outputs=$corpora/train.spm.$lang
			done
	else
		echo "Train already segmented"  >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi

	if [[ ! -f $EXPERIMENT_DIR/corpora/valid.spm.$l1 ]]; then
		for lang in $l1 $l2
		do 
			python $TOOLS/spm_encode.py --model $spm_model_dir/spm.model --output_format=piece --inputs=$EXPERIMENT_DIR/corpora/valid.$lang --outputs=$EXPERIMENT_DIR/corpora/valid.spm.$lang
			python $TOOLS/spm_encode.py --model $spm_model_dir/spm.model --output_format=piece --inputs=$EXPERIMENT_DIR/corpora/test.$lang --outputs=$EXPERIMENT_DIR/corpora/test.spm.$lang
		done
	else
		echo "Valid and Test already segmented"  >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi
}

function preprocess() {
	local l1=$1
	local l2=$2
	local model=$3
	local dir=$4
	local mono=$5
	if [[ ! -d $dir/data_bin ]]; then
		bash script_preprocess.sh --source=$l1 --target=$l2 --model-dir=$model --experiment-dir=$dir --mono=$mono #1=mono, 0=train,valid,test
	else
		echo "Preprocess already executed"  >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi
}

function fine_tuning() {
	local l1=$1
	local l2=$2
	local model=$3
	local dir=$4
	if [[ ! -d $dir/checkpoint ]]; then
		bash script_train.sh --source=$l1 --target=$l2 --model-dir=$model --experiment-dir=$dir --model-type="mbart50"
	else
		echo "Train already done" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi
}

# --interactive=1: fairseq-interactive
# --interactive=0: fairseq-generate
function translate() {
	local l1=$1
	local l2=$2
	local model=$3
	local dir=$4
	local INTERACTIVE=$5
	local gen=$6

	if [[ ! -f $dir/$gen.out.$l2 ]]; then
		bash script_generate.sh --source=$l1 --target=$l2 --model-dir=$model --experiment-dir=$dir --model-type="mbart50" --interactive=$INTERACTIVE --gen-subset=$gen
	else
		echo "$gen already translated" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi
}

function BLEU_validation() {
	local ref=$1
	local output=$2
	local dir=$3

	cat $dir/$output | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $dir/$output.clean.sys
	python $TOOLS/sacrebleu/sacrebleu/sacrebleu.py $ref < $dir/$output.clean.sys --metrics bleu chrf | cut -f 3 -d  ' ' > $dir/$output.BLEU_chrf_spBLEU
	python $TOOLS/sacrebleu/sacrebleu/sacrebleu.py $ref < $dir/$output.clean.sys --tokenize spm | cut -f 3 -d  ' ' >> $dir/$output.BLEU_chrf_spBLEU
}

function translate_monolingual_corpora() {
	local l1=$1
	local l2=$2
	local model=$3
	local corpus=$4
	local dir=$5
	local output=$6

	local spm_model_dir=

	mkdir -p $dir/spm

	if [ $l1 = "en_XX" ]; then
		spm_model_dir=$MBART50_1N
	else
		spm_model_dir=$MBART50_N1
	fi

	if [[ ! -f $dir/$output.$l2.synthetic ]] ; then
		python $TOOLS/spm_encode.py --model $spm_model_dir/spm.model --output_format=piece --inputs=$corpus --outputs=$dir/spm/$output.spm.$l1
		awk -F ' ' '{if (NF <= 100){ print $0; } }' $dir/spm/$output.spm.$l1 > $dir/spm/$output.spm.clean.$l1
		rm $dir/spm/$output.spm.$l1
		mv $dir/spm/$output.spm.clean.$l1 $dir/spm/$output.spm.$l1

		local lines=$(awk 'END{print NR}' $dir/spm/$output.spm.$l1) #Make fake corpus for preprocess and generate
		seq 1 $lines > $dir/spm/$output.spm.$l2
		
		echo "call preprocess for mono" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
		bash script_preprocess.sh --source=$l1 --target=$l2 --model-dir=$model --experiment-dir=$dir --mono=1
		
		for lang in $l1 $l2; do
			mv $dir/data_bin_mono/valid.$l1-$l2.$lang.bin $dir/data_bin_mono/mono.$l1-$l2.$lang.bin
			mv $dir/data_bin_mono/valid.$l1-$l2.$lang.idx $dir/data_bin_mono/mono.$l1-$l2.$lang.idx
		done

		echo "call generate for mono" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
		translate $l1 $l2 $model $dir 0 "mono"

		python $TOOLS/spm_decode.py --model $spm_model_dir/spm.model --input_format=piece --input=$dir/spm/$output.spm.$l1 > $dir/$output.$l1.original #spm_decode spm/original
		
		cat $dir/$output.out.$l2 | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $dir/$output.$l2.synthetic #create synthetic corpus from output
	else
		echo "Monolingual already translated" >> $EXPERIMENT_DIR/$LANG_1-$LANG_2.log
	fi
}

function train_and_validate () {
	local source_lang=$1
	local target_lang=$2
	local iteration=$3
	local mbart50_type=$4 #n1 or 1n
	local mono_synthetic=$5
	local mono_original=$6

	local model_to_train=
	if [ $source_lang = "en_XX" ]; then
		model_to_train=$BEST_1N
	else
		model_to_train=$BEST_N1
	fi

	prepare_dir $iteration $mbart50_type $source_lang $target_lang $mono_synthetic $mono_original
	spm_encode $source_lang $target_lang $MBART50_DIR/$mbart50_type $EXPERIMENT_DIR/it-$iteration-$mbart50_type/corpora
	preprocess $source_lang $target_lang $MBART50_DIR/$mbart50_type $EXPERIMENT_DIR/it-$iteration-$mbart50_type 0
	fine_tuning $source_lang $target_lang $model_to_train $EXPERIMENT_DIR/it-$iteration-$mbart50_type
	translate $source_lang $target_lang $EXPERIMENT_DIR/it-$iteration-$mbart50_type/checkpoint $EXPERIMENT_DIR/it-$iteration-$mbart50_type 0 "valid"
	BLEU_validation $EXPERIMENT_DIR/corpora/valid.$target_lang valid.out.$target_lang $EXPERIMENT_DIR/it-$iteration-$mbart50_type
	local BLEU=$(head -1 $EXPERIMENT_DIR/it-$iteration-$mbart50_type/valid.out.$target_lang.BLEU_chrf_spBLEU)
	echo $BLEU
}

function first_fine_tune () {
	local source_lang=$1
	local target_lang=$2
	local mbart50_type=$3 #n1 or 1n

	echo "#### $mbart50_type ####"
	
	local new_BLEU=$(train_and_validate $source_lang $target_lang first $mbart50_type null null)
	
	mv $EXPERIMENT_DIR/it-first-$mbart50_type/valid.out.$target_lang.BLEU_chrf_spBLEU $EXPERIMENT_DIR/it-first-$mbart50_type/fine-tuned-valid.out.$target_lang.BLEU_chrf_spBLEU
	mv $EXPERIMENT_DIR/it-first-$mbart50_type/valid.out.$target_lang.clean.sys  $EXPERIMENT_DIR/it-first-$mbart50_type/fine-tuned-valid.out.$target_lang.clean.sys
	rm $EXPERIMENT_DIR/it-first-$mbart50_type/valid.out.$target_lang
	
	translate $source_lang $target_lang $MBART50_DIR/$mbart50_type $EXPERIMENT_DIR/it-first-$mbart50_type 0 "valid"
	BLEU_validation $EXPERIMENT_DIR/corpora/valid.$target_lang valid.out.$target_lang $EXPERIMENT_DIR/it-first-$mbart50_type
	local original_BLEU=$(head -1 $EXPERIMENT_DIR/it-first-$mbart50_type/valid.out.$target_lang.BLEU_chrf_spBLEU)

	if awk "BEGIN {exit !($new_BLEU >= $original_BLEU)}"; then
        echo "$source_lang-$target_lang improved from $original_BLEU to $new_BLEU"
		head -1000000 $CORPORA_DIR/mono.$source_lang > $EXPERIMENT_DIR/it-first-$mbart50_type/mono.$source_lang
		if [ $source_lang = "en_XX" ]; then
			BEST_BLEU_1N=$new_BLEU
			BEST_1N=$EXPERIMENT_DIR/it-first-$mbart50_type/checkpoint
			echo "Translate monolingual $source_lang corpora"
			translate_monolingual_corpora  $source_lang $target_lang $BEST_1N $EXPERIMENT_DIR/it-first-$mbart50_type/mono.$source_lang $EXPERIMENT_DIR/it-first-$mbart50_type "mono"
			mono_1_original=$EXPERIMENT_DIR/it-first-$mbart50_type/mono.$source_lang.original
			mono_n_synthetic=$EXPERIMENT_DIR/it-first-$mbart50_type/mono.$target_lang.synthetic
		else
			BEST_BLEU_N1=$new_BLEU
			BEST_N1=$EXPERIMENT_DIR/it-first-$mbart50_type/checkpoint
			echo "Translate monolingual $source_lang corpora"
			translate_monolingual_corpora  $source_lang $target_lang $BEST_N1 $EXPERIMENT_DIR/it-first-$mbart50_type/mono.$source_lang $EXPERIMENT_DIR/it-first-$mbart50_type "mono"
			mono_n_original=$EXPERIMENT_DIR/it-first-$mbart50_type/mono.$source_lang.original
			mono_1_synthetic=$EXPERIMENT_DIR/it-first-$mbart50_type/mono.$target_lang.synthetic
		fi
	else
        	echo "$mbart50_type NOT improved"
    fi
}

function make_synthetic_corpus () {

	local model_dir=$1
	local score=$2
	local score_number=${score:0:1}.${score:1:2}
	local source_target=$3
	local s_t=(${source_target//_/ })

	LC_NUMERIC=POSIX awk -F "\t" '$3 > '$score_number' { print $1 "\t" $2 }' $model_dir/$source_target.$sub_lang_1-$sub_lang_2.classified > $model_dir/$source_target.$score.$sub_lang_1-$sub_lang_2
	cut -f1 $model_dir/$source_target.$score.$sub_lang_1-$sub_lang_2 > $model_dir/${s_t[0]}.$score.$LANG_1
	cut -f2 $model_dir/$source_target.$score.$sub_lang_1-$sub_lang_2 > $model_dir/${s_t[1]}.$score.$LANG_2
	rm $model_dir/$source_target.$score.$sub_lang_1-$sub_lang_2
}

function find_best_bicleaner_score () {

	local best_local_bleu_1n=0.0
	local best_local_bleu_n1=0.0

	local best_local_model_1n=""
	local best_local_model_n1=""

   	scores="00 01 02 03 04 05 06 07 08 09"
   	./classify-corpus.sh $sub_lang_1 $sub_lang_2 $EXPERIMENT_DIR $mono_1_original $mono_n_synthetic $BEST_1N/../original_synthetic
   	./classify-corpus.sh $sub_lang_1 $sub_lang_2 $EXPERIMENT_DIR $mono_1_synthetic $mono_n_original $BEST_N1/../synthetic_original
	
	for score in $scores ; do
		make_synthetic_corpus $BEST_N1/.. $score synthetic_original
		local new_BLEU_1n=$(train_and_validate $LANG_1 $LANG_2 score_$score mbart50-1n $BEST_N1/../synthetic.$score.$LANG_1	$BEST_N1/../original.$score.$LANG_2)
		if awk "BEGIN {exit !($new_BLEU_1n >= $best_local_bleu_1n)}"; then
			best_local_bleu_1n=$new_BLEU_1n
			best_local_model_1n=$EXPERIMENT_DIR/it-score_$score-mbart50-1n/checkpoint
			BICLEANER_SCORE_1N=$score
		fi

		make_synthetic_corpus $BEST_1N/.. $score original_synthetic
		local new_BLEU_n1=$(train_and_validate $LANG_2 $LANG_1 score_$score mbart50-n1 $BEST_1N/../synthetic.$score.$LANG_2 $BEST_1N/../original.$score.$LANG_1)
		if awk "BEGIN {exit !($new_BLEU_n1 >= $best_local_bleu_n1)}"; then
			best_local_bleu_n1=$new_BLEU_n1
			best_local_model_n1=$EXPERIMENT_DIR/it-score_$score-mbart50-n1/checkpoint
			BICLEANER_SCORE_N1=$score
		fi
	done

	if awk "BEGIN {exit !($best_local_bleu_1n >= $BEST_BLEU_1N)}"; then
		BEST_BLEU_1N=$best_local_bleu_1n
		BEST_1N=$best_local_model_1n
	fi

	if awk "BEGIN {exit !($best_local_bleu_n1 >= $BEST_BLEU_N1)}"; then
		BEST_BLEU_N1=$best_local_bleu_n1
		BEST_N1=$best_local_model_n1
	fi
}

function iterative-Backtranslation () {
	
	local en_n_improved=true
    local n_en_improved=true

	echo "Translate monolingual $LANG_2 corpora"
	head -2000000 $CORPORA_DIR/mono.$LANG_2 > $BEST_N1/../mono.$LANG_2
	translate_monolingual_corpora $LANG_2 $LANG_1 $BEST_N1 $BEST_N1/../mono.$LANG_2 $BEST_N1/.. "mono"
	./classify-corpus.sh $sub_lang_1 $sub_lang_2 $EXPERIMENT_DIR $mono_1_synthetic $mono_n_original $BEST_N1/../synthetic_original
	make_synthetic_corpus $BEST_N1/.. $BICLEANER_SCORE_1N synthetic_original
	mono_n_original=$BEST_N1/../original.$BICLEANER_SCORE_1N.$LANG_2
	mono_1_synthetic=$BEST_N1/../synthetic.$BICLEANER_SCORE_1N.$LANG_1

    for IT in $(seq 8) ; do
        echo "Iteration $IT"
		
		local new_BLEU_1n=$(train_and_validate $LANG_1 $LANG_2 $IT mbart50-1n $mono_1_synthetic $mono_n_original)

        if awk "BEGIN {exit !($new_BLEU_1n >= $BEST_BLEU_1N)}"; then
            echo "$LANG_1-$LANG_2 improved from $BEST_BLEU_1N to $new_BLEU_1n"
            BEST_BLEU_1N=$new_BLEU_1n
			BEST_1N=$EXPERIMENT_DIR/it-$IT-mbart50-1n/checkpoint
            en_n_improved=true

			head -$(( $IT+2 ))000000 $CORPORA_DIR/mono.$LANG_1 > $EXPERIMENT_DIR/it-$IT-mbart50-1n/mono.$LANG_1
			if [[ ! -f $EXPERIMENT_DIR/it-$IT-mbart50-1n/mono.$LANG_2.synthetic ]]; then
				translate_monolingual_corpora $LANG_1 $LANG_2 $BEST_1N $EXPERIMENT_DIR/it-$IT-mbart50-1n/mono.$LANG_1 $EXPERIMENT_DIR/it-$IT-mbart50-1n "mono"
			fi
			mono_1_original=$EXPERIMENT_DIR/it-$IT-mbart50-1n/mono.$LANG_1.original
			mono_n_synthetic=$EXPERIMENT_DIR/it-$IT-mbart50-1n/mono.$LANG_2.synthetic
			
			echo "Classify corpus"
			./classify-corpus.sh $sub_lang_1 $sub_lang_2 $EXPERIMENT_DIR $mono_1_original $mono_n_synthetic $EXPERIMENT_DIR/it-$IT-mbart50-1n/original_synthetic
			make_synthetic_corpus $BEST_1N/.. $BICLEANER_SCORE_N1 original_synthetic
			mono_1_original=$EXPERIMENT_DIR/it-$IT-mbart50-1n/original.$BICLEANER_SCORE_N1.$LANG_1
			mono_n_synthetic=$EXPERIMENT_DIR/it-$IT-mbart50-1n/synthetic.$BICLEANER_SCORE_N1.$LANG_2
        else
            echo "$LANG_1-$LANG_2 NOT improved"
            en_n_improved=false
        fi

        local new_BLEU_n1=$(train_and_validate $LANG_2 $LANG_1 $IT mbart50-n1 $mono_n_synthetic $mono_1_original)
		if awk "BEGIN {exit !($new_BLEU_n1 >= $BEST_BLEU_N1)}"; then
            echo "$LANG_2-$LANG_1 improved from $BEST_BLEU_N1 to $new_BLEU_n1"
            BEST_BLEU_N1=$new_BLEU_n1
			BEST_N1=$EXPERIMENT_DIR/it-$IT-mbart50-n1/checkpoint
            n_en_improved=true

			head -$(( $IT+2 ))000000 $CORPORA_DIR/mono.$LANG_2 > $EXPERIMENT_DIR/it-$IT-mbart50-n1/mono.$LANG_2
			if [[ ! -f $EXPERIMENT_DIR/it-$IT-mbart50-n1/mono.$LANG_1.synthetic ]]; then
				translate_monolingual_corpora $LANG_2 $LANG_1 $BEST_N1 $EXPERIMENT_DIR/it-$IT-mbart50-n1/mono.$LANG_2 $EXPERIMENT_DIR/it-$IT-mbart50-n1 "mono"
			fi
        	echo "mono.$LANG_2 translated"
			mono_n_original=$EXPERIMENT_DIR/it-$IT-mbart50-n1/mono.$LANG_2.original
			mono_1_synthetic=$EXPERIMENT_DIR/it-$IT-mbart50-n1/mono.$LANG_1.synthetic

			echo "Classify corpus"
			./classify-corpus.sh $sub_lang_1 $sub_lang_2 $EXPERIMENT_DIR $mono_1_synthetic $mono_n_original $EXPERIMENT_DIR/it-$IT-mbart50-n1/synthetic_original
			make_synthetic_corpus $BEST_N1/.. $BICLEANER_SCORE_1N synthetic_original
			mono_n_original=$EXPERIMENT_DIR/it-$IT-mbart50-n1/original.$BICLEANER_SCORE_1N.$LANG_2
			mono_1_synthetic=$EXPERIMENT_DIR/it-$IT-mbart50-n1/synthetic.$BICLEANER_SCORE_1N.$LANG_1
		else
            echo "$LANG_2-$LANG_1 NOT improved"
            n_en_improved=false
		fi
        
		if [[ "$en_n_improved" = false && "$n_en_improved" = false ]]; then
			echo "None of the directions improved: finishing"
			## Make directories with synthetic corpora
			mkdir $EXPERIMENT_DIR/synthetic_corpora
			cp $mono_1_synthetic $EXPERIMENT_DIR/synthetic_corpora/mono_1_synthetic
			cp $mono_n_original $EXPERIMENT_DIR/synthetic_corpora/mono_n_original
			cp $mono_n_synthetic $EXPERIMENT_DIR/synthetic_corpora/mono_n_synthetic
			cp $mono_1_original $EXPERIMENT_DIR/synthetic_corpora/mono_1_original

			echo "### Best result N1 ###" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best BLEU = $BEST_BLEU_N1" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best model = $BEST_N1" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best score = $BICLEANER_SCORE_N1" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "### Best result 1N ###" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best BLEU = $BEST_BLEU_1N" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best model = $BEST_1N" >> $EXPERIMENT_DIR/fine_tuning.result
			echo "Best score = $BICLEANER_SCORE_1N" >> $EXPERIMENT_DIR/fine_tuning.result

			exit 1
		fi
    done
}

function main () {
	first_fine_tune $LANG_1 $LANG_2 "mbart50-1n"
	first_fine_tune $LANG_2 $LANG_1 "mbart50-n1"
	
	if [[ $mono_n_original = "" ]] || [[ $mono_1_original = "" ]]; then
		echo "mBART50 NOT improved with parallel corpus"
	else
		find_best_bicleaner_score
		iterative-Backtranslation
	fi
}

main

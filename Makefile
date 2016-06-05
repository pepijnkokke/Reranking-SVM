DIR=/Users/pepijn/Projects/nlp2

default:\
	$(DIR)/docs/svm-100-table.tex\
	$(DIR)/docs/svm-100-table-500.tex\
	$(DIR)/docs/svm-100-table-1000.tex\
	$(DIR)/docs/svm-500-table.tex\
	$(DIR)/docs/svm-2900-table.tex\

$(DIR)/docs/svm-100-table.tex:
	multeval eval\
		--meteor.language de\
		--refs		$(wildcard $(DIR)/data/nlp2-test.first2100.de)\
		--hyps-baseline $(wildcard $(DIR)/out/*-baseline.out)\
		--hyps-sys1	$(wildcard $(DIR)/out/*-svm-100-basic.out)\
		--hyps-sys2	$(wildcard $(DIR)/out/*-svm-100-pos.out)\
		--hyps-sys3	$(wildcard $(DIR)/out/*-svm-100-pos-bigrams.out)\
		--hyps-sys4	$(wildcard $(DIR)/out/*-svm-100-ex-pos.out)\
		--hyps-sys5	$(wildcard $(DIR)/out/*-svm-100-ex-pos-bigrams.out)\
		--hyps-sys6	$(wildcard $(DIR)/out/*-svm-100-combinations.out)\
		--hyps-sys7	$(wildcard $(DIR)/out/*-svm-100-embedding.out)\
		--hyps-sys8	$(wildcard $(DIR)/out/*-svm-100-full.out)\
		--latex		$(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 1/basic          /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 2/pos            /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 3/pos bi.\\      /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 4/ex.\\ pos      /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 5/ex.\\ pos bi.\\/' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 6/combination    /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 7/embedding      /' $(DIR)/docs/svm-100-table.tex
	sed -i.bak 's/system 8/full           /' $(DIR)/docs/svm-100-table.tex

$(DIR)/docs/svm-500-table.tex:
	multeval eval\
		--meteor.language de\
		--refs		$(wildcard $(DIR)/data/nlp2-test.first2100.de)\
		--hyps-baseline $(wildcard $(DIR)/out/*-baseline.out)\
		--hyps-sys1	$(wildcard $(DIR)/out/*-svm-500-basic.out)\
		--hyps-sys2	$(wildcard $(DIR)/out/*-svm-500-pos.out)\
		--hyps-sys3	$(wildcard $(DIR)/out/*-svm-500-pos-bigrams.out)\
		--hyps-sys4	$(wildcard $(DIR)/out/*-svm-500-ex-pos.out)\
		--hyps-sys5	$(wildcard $(DIR)/out/*-svm-500-ex-pos-bigrams.out)\
		--hyps-sys6	$(wildcard $(DIR)/out/*-svm-500-combinations.out)\
		--hyps-sys7	$(wildcard $(DIR)/out/*-svm-500-embedding.out)\
		--hyps-sys8	$(wildcard $(DIR)/out/*-svm-500-full.out)\
		--latex		$(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 1/basic          /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 2/pos            /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 3/pos bi.\\      /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 4/ex.\\ pos      /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 5/ex.\\ pos bi.\\/' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 6/combination    /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 7/embedding      /' $(DIR)/docs/svm-500-table.tex
	sed -i.bak 's/system 8/full           /' $(DIR)/docs/svm-500-table.tex

$(DIR)/docs/svm-2900-table.tex:
	multeval eval\
		--meteor.language de\
		--refs		$(wildcard $(DIR)/data/nlp2-test.first2100.de)\
		--hyps-baseline $(wildcard $(DIR)/out/*-baseline.out)\
		--hyps-sys1	$(wildcard $(DIR)/out/*-svm-2900-basic.out)\
		--hyps-sys2	$(wildcard $(DIR)/out/*-svm-2900-pos.out)\
		--hyps-sys3	$(wildcard $(DIR)/out/*-svm-2900-pos-bigrams.out)\
		--hyps-sys4	$(wildcard $(DIR)/out/*-svm-2900-ex-pos.out)\
		--hyps-sys5	$(wildcard $(DIR)/out/*-svm-2900-ex-pos-bigrams.out)\
		--hyps-sys6	$(wildcard $(DIR)/out/*-svm-2900-combinations.out)\
		--hyps-sys7	$(wildcard $(DIR)/out/*-svm-2900-embedding.out)\
		--hyps-sys8	$(wildcard $(DIR)/out/*-svm-2900-full.out)\
		--latex		$(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 1/basic          /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 2/pos            /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 3/pos bi.\\      /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 4/ex.\\ pos      /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 5/ex.\\ pos bi.\\/' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 6/combination    /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 7/embedding      /' $(DIR)/docs/svm-2900-table.tex
	sed -i.bak 's/system 8/full           /' $(DIR)/docs/svm-2900-table.tex

$(DIR)/docs/svm-100-table-500.tex:
	multeval eval\
		--meteor.language de\
		--refs		$(wildcard $(DIR)/data/nlp2-test.first2100.de)\
		--hyps-baseline $(wildcard $(DIR)/out/*-baseline.out)\
		--hyps-sys1	$(wildcard $(DIR)/out/*-svm-100-basic-500.out)\
		--hyps-sys2	$(wildcard $(DIR)/out/*-svm-100-pos-500.out)\
		--latex		$(DIR)/docs/svm-100-table-500.tex
	sed -i.bak 's/system 1/basic   /' $(DIR)/docs/svm-100-table-500.tex
	sed -i.bak 's/system 2/pos     /' $(DIR)/docs/svm-100-table-500.tex

$(DIR)/docs/svm-100-table-1000.tex:
	multeval eval\
		--meteor.language de\
		--refs		$(wildcard $(DIR)/data/nlp2-test.first2100.de)\
		--hyps-baseline $(wildcard $(DIR)/out/*-baseline.out)\
		--hyps-sys1	$(wildcard $(DIR)/out/*-svm-100-basic-1000.out)\
		--hyps-sys2	$(wildcard $(DIR)/out/*-svm-100-pos-1000.out)\
		--latex		$(DIR)/docs/svm-100-table-1000.tex
	sed -i.bak 's/system 1/basic   /' $(DIR)/docs/svm-100-table-1000.tex
	sed -i.bak 's/system 2/pos     /' $(DIR)/docs/svm-100-table-1000.tex

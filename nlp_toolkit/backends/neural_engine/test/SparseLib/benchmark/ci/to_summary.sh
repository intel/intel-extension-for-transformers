cat $1 |
    gawk 'BEGIN {
        total_labels = 0;
        total_cases = 0;
    }
    {
        if ($1 == ">>>>>>>###") {
            for (i=2; i<=NF; i++) {
                curr_label[i-1] = $i;
                if (!($i in labels_dict)) labels_dict[$i]=(++total_labels);
            } 
        } else if ($1 == ">>>>>>>>>>") {
            total_cases ++;
            for (i=2; i<=NF; i++) {
                result[total_cases,"label",curr_label[i-1]] = $i;
            }
            getline out_acc;
            result[total_cases,"acc"] = out_acc == "result correct" ? "correct" : "incorrect";

            getline out_perf;
            result[total_cases,"perf"] = out_perf;
        } 
    } END {
        PROCINFO["sorted_in"]="@val_num_asc";
        for (label in labels_dict) {
            printf("%s;", label);
        }
        printf("acc;perf;\n")
        for (i_case=1; i_case <= total_cases; i_case++) {
            for (label in labels_dict) {
                if ((i_case,"label",label) in result) 
                    printf("%s;", result[i_case,"label",label]);
                else
                    printf("na;");
            }
            if ((i_case, "acc") in result)
                printf("%s;", result[i_case, "acc"]);
            else
                printf("%s;\n", "na");
            if ((i_case, "perf") in result)
                printf("%s;\n", result[i_case, "perf"]);
            else
                printf("%s;\n", "na");
        }
    }'

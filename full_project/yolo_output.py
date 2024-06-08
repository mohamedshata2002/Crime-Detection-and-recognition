def output_list(results,names,conf) :
    output_name = []
    for m in results[0].boxes:
            if m.conf>=conf:
             output_list.append(m.cls)
    output = list(set(output_list))
    return output
            
           
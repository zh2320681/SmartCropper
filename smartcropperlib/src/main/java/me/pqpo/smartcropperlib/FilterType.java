package me.pqpo.smartcropperlib;

/**
 * @author shrek
 * @date: 2021-03-25
 */
public enum FilterType {
    enhance, blackWhite, brighten, grey, soft;

    public static FilterType valueOf(int index){
        if(index == 0){
            return FilterType.enhance;
        } else if (index == 1){
            return FilterType.blackWhite;
        } else if (index == 2){
            return FilterType.brighten;
        } else if (index == 3){
            return FilterType.grey;
        } else if (index == 4){
            return FilterType.soft;
        }

        return FilterType.grey;
    }
}

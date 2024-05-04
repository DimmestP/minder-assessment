import pandas as pd
from statsmodels.tsa.api import VAR

def count_events_per_interval(data, start: str, end: str, interval: str):
    """
    """
    
    counted_events = pd.DataFrame({"datetime":pd.date_range(start=start, end=end, freq=interval, tz = 'UTC'),
                                  "event_count" : 0})

    for x in range(len(counted_events) - 1):
        
        events_in_interval = data[ data["datetime"] >  counted_events["datetime"][x]]
        events_in_interval = events_in_interval[ events_in_interval["datetime"] <= counted_events["datetime"][x + 1]]
        counted_events.loc[x,"event_count"] = len(events_in_interval)
        
    # Return dataframe without last entry as it is always zero (used as an upper limit to sum events on 31st)
    return counted_events[0:len(counted_events) - 1] 

def fit_VAR_model(data, lag: int, freq: str):
    """
    """
    data = data.pivot_table(index="datetime",values="event_count", columns="location")
    model = VAR(data, freq=freq)
    results = model.fit(lag)
    
    return results.params
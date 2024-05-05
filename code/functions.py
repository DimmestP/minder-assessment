import pandas as pd
from statsmodels.tsa.api import VAR

def count_events_per_interval(data, start: str, end: str, interval: str):
    """
    Function to count all of the detected motion events in the intervals across the defined time period.

    Parameters
    ----------

    data : DataFrame
        motion dataframe with columns: ["event_id", "datetime", "location"].
    start : str
        Left bound for generating dates.
    end : str
        Right bound for generating dates.
    interval : str
        Size of intervals to break date range into, e.g. ‘5h’.

    Returns
    -------
    DataFrame
        Counts per interval per location and house
    
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
    Function to fit vector autoregression model across all locations with count data within a home.

    Parameters
    ----------

    data : DataFrame
        event_counts dataframe with columns: ["event_count", "datetime", "location"].
    lag : int
        Number of previous values to use to predect the next in the autoregression model.
    freq : str
        The unit of time between measurements, e.g. 'D' for a day.

    Returns
    -------
    DataFrame
        VAR parameters for a home
    
    """
    data = data.pivot_table(index="datetime",values="event_count", columns="location")
    model = VAR(data, freq=freq)
    results = model.fit(lag)
    
    return results.params

def preprocess_events_data(data, start: str = "2024-01-01", end: str = "2024-02-01", 
                           interval: str = "1d", lag: int = 2, freq: str = "D"):
    """
    Wrapper function to ingest raw motion data, count events, fit VAR model and prepare for classification.

    Parameters
    ----------

    data : DataFrame
        motion dataframe with columns: ["event_id", "home_id", "datetime", "location"].
    start : str
        Left bound for generating dates.
    end : str
        Right bound for generating dates.
    interval : str
        Size of intervals to break date range into, e.g. ‘5h’.
    lag : int
        Number of previous values to use to predect the next in the autoregression model.
    freq : str
        The unit of time between measurements, e.g. 'D' for a day.

    Returns
    -------
    DataFrame
        VAR parameters and binary location results
        
    """
    # Binary variables
    homes_with_conservatory = data[data.location.str.contains('conservatory')][["home_id"]].groupby(["home_id"]).head(1)

    if len(homes_with_conservatory) > 0: homes_with_conservatory["has_conservatory"] = True

    else:
        homes_with_conservatory = data[["home_id"]].groupby(["home_id"]).head(1)
        homes_with_conservatory["has_conservatory"] = False

    homes_with_dining_room = data[data.location.str.contains('dining room')][["home_id"]].groupby(["home_id"]).head(1)

    if len(homes_with_dining_room) > 0: homes_with_dining_room["has_dining_room"] = True

    else:
        homes_with_dining_room = data[["home_id"]].groupby(["home_id"]).head(1)
        homes_with_dining_room["has_dining_room"] = False
        
    
    homes_with_study = data[data.location.str.contains('study')][["home_id"]].groupby(["home_id"]).head(1)

    if len(homes_with_study) > 0: homes_with_study["has_study"] = True

    else:
        homes_with_study = data[["home_id"]].groupby(["home_id"]).head(1)
        homes_with_study["has_study"] = False

    homes_with_binary_locations = homes_with_conservatory.set_index("home_id")

    homes_with_binary_locations = homes_with_binary_locations.join(homes_with_dining_room.set_index("home_id"), how = "outer")

    homes_with_binary_locations = homes_with_binary_locations.join(homes_with_study.set_index("home_id"), how = "outer")

    homes_with_binary_locations = homes_with_binary_locations.fillna(False)
    
    # Time series variables
    events_selected_locations = data[data.location.str.contains('bedroom1|lounge|bathroom1|hallway|kitchen')]
    
    counted_events_per_house = events_selected_locations.groupby(["home_id","location"]).apply(count_events_per_interval,
                                                                                               start, end, interval)

    # Tidy up dataframe index
    counted_events_per_house = counted_events_per_house.reset_index(level=['home_id','location']).reset_index(drop=True)

    modelled_event_counts = counted_events_per_house.groupby("home_id").apply(fit_VAR_model, lag, freq)

    # Move index to column names
    modelled_event_counts = modelled_event_counts.reset_index(level='home_id')
    modelled_event_counts["coefficient"] = modelled_event_counts.index
    modelled_event_counts = modelled_event_counts.reset_index(drop=True)
    
    # Pivot all coefficients to column values and flatten column name hierchy 
    modelled_event_counts = modelled_event_counts.pivot(columns="coefficient", index="home_id")
    modelled_event_counts.columns = [' '.join(col).strip() for col in modelled_event_counts.columns.values]
    
    # Set missing vector autoregression coefficients to 0
    modelled_event_counts = modelled_event_counts.fillna(0)
    
    # Join results with binary data to prepare for classification
    modelled_event_counts = modelled_event_counts.join(homes_with_binary_locations).fillna(False)

    return modelled_event_counts
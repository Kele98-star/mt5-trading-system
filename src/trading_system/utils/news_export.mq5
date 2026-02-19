//+------------------------------------------------------------------+
//|                         CalendarNewsExporterAuto.mq5              |
//|             Auto-export calendar events daily via timer           |
//+------------------------------------------------------------------+
#property copyright "Kele98-Star"
#property version   "1.0"
#property strict

input int  DaysBack             = 1;              // Days back from today
input int  DaysForward          = 3;              // Days forward from today
input bool IncludeHolidays      = true;           // Include market holidays
input int  TimerIntervalSeconds = 3600 * 4;       // Timer check interval (4 hour default)
input bool ExcludeLowImpact     = true;           // Exclude LOW-importance events (non-holidays)

static datetime g_last_export_date = 0;

datetime DateFloor(datetime t)
{
   return (t / 86400) * 86400;
}

// Forward declarations
bool ExportCalendarData();
bool CheckAndExport(const bool force);

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!EventSetTimer(TimerIntervalSeconds))
   {
      Print("Failed to set timer. Error: ", GetLastError());
      return INIT_FAILED;
   }

   Print("Calendar Auto-Exporter initialized. Timer interval: ", TimerIntervalSeconds, "s");

   // Force export on initialization (writes/overwrites calendar.csv)
   CheckAndExport(true);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("Calendar Auto-Exporter stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Timer function - checks for new day and triggers export          |
//+------------------------------------------------------------------+
void OnTimer()
{
   CheckAndExport(false);
}

//+------------------------------------------------------------------+
//| Shared gate: runs on init and timer                              |
//+------------------------------------------------------------------+
bool CheckAndExport(const bool force)
{
   datetime now          = TimeTradeServer();
   datetime current_date = DateFloor(now);

   if(force || g_last_export_date == 0 || current_date > g_last_export_date)
   {
      Print("Export trigger (force=", (force ? "true" : "false"), "). Executing calendar export...");

      if(ExportCalendarData())
      {
         g_last_export_date = current_date;
         return true;
      }

      Print("Export failed. Will retry on next timer tick.");
      return false;
   }

   return true; // No export needed
}

//+------------------------------------------------------------------+
//| Core export logic                                                |
//| Returns true on success, false on failure                        |
//+------------------------------------------------------------------+
bool ExportCalendarData()
{
   datetime now = TimeTradeServer();

   datetime start_time = now - (datetime)DaysBack * 86400;
   datetime end_time   = now + (datetime)DaysForward * 86400;

   MqlCalendarValue values[];
   if(!CalendarValueHistory(values, start_time, end_time, NULL, NULL))
   {
      Print("Failed to retrieve calendar data. Error: ", GetLastError());
      return false;
   }

   int total = ArraySize(values);
   if(total == 0)
   {
      Print("No calendar events found in range");
      return false;
   }

   string filename = "calendar.txt";

   // Overwrite snapshot each export
   int file_handle = FileOpen(filename, FILE_WRITE | FILE_TXT, CP_UTF8);
   if(file_handle == INVALID_HANDLE)
   {
      Print("Failed to create file. Error: ", GetLastError());
      return false;
   }

   FileWriteString(file_handle,"Date,Time,Currency,Event,Impact,Type\r\n");



   int event_count   = 0;
   int holiday_count = 0;
   int excluded_low  = 0;

   for(int i = 0; i < total; i++)
   {
      MqlCalendarEvent   event;
      MqlCalendarCountry country;

      if(!CalendarEventById(values[i].event_id, event))
         continue;

      if(!CalendarCountryById(event.country_id, country))
         continue;

      // Holiday filter
      if(!IncludeHolidays && event.type == CALENDAR_TYPE_HOLIDAY)
         continue;

      // Low-importance filter (only for non-holidays)
      if(ExcludeLowImpact && event.type != CALENDAR_TYPE_HOLIDAY && event.importance == CALENDAR_IMPORTANCE_LOW)
      {
         excluded_low++;
         continue;
      }

      string event_type = "UNKNOWN";
      switch(event.type)
      {
         case CALENDAR_TYPE_EVENT:     event_type = "EVENT";     break;
         case CALENDAR_TYPE_INDICATOR: event_type = "INDICATOR"; break;
         case CALENDAR_TYPE_HOLIDAY:   event_type = "HOLIDAY";   break;
         default: break;
      }

      if(event.type == CALENDAR_TYPE_HOLIDAY) holiday_count++;
      else                                    event_count++;

      string event_date = TimeToString(values[i].time, TIME_DATE);
      string event_time = TimeToString(values[i].time, TIME_MINUTES);

      string impact = "NONE";
      switch(event.importance)
      {
         case CALENDAR_IMPORTANCE_LOW:      impact = "LOW";    break;
         case CALENDAR_IMPORTANCE_MODERATE: impact = "MEDIUM"; break;
         case CALENDAR_IMPORTANCE_HIGH:     impact = "HIGH";   break;
         default: break;
      }

      // Explicitly wrap the event name in double quotes to prevent delimiter collision
      string event_name_safe = "\"" + event.name + "\"";
      
      string line =
      event_date + "," +
      event_time + "," +
      country.currency + "," +
      "\"" + event.name + "\"," +
      impact + "," +
      event_type + "\r\n";
   
   FileWriteString(file_handle, line);


   }

   FileClose(file_handle);

   Print("========================================");
   Print("Calendar export completed successfully:");
   Print("  Events: ", event_count);
   Print("  Holidays: ", holiday_count);
   Print("  Excluded LOW: ", excluded_low);
   Print("  Total written: ", event_count + holiday_count);
   Print("  Date range: ", TimeToString(start_time, TIME_DATE), " to ", TimeToString(end_time, TIME_DATE));
   Print("  File: ", filename);
   Print("  Path: ", TerminalInfoString(TERMINAL_DATA_PATH), "\\MQL5\\Files\\", filename);
   Print("========================================");

   return true;
}



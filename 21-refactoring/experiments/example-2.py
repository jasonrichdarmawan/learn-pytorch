# %%

import collections
import bisect

DELETED = object()

class InMemoryDBImpl:
    
    def __init__(self):
        """
        Structure: { key: { field: [(timestamp, value, epxiration_ts), ...] } }
        value can be an int or DELETED
        """
        # self.records = collections.defaultdict(dict)
        self.records = collections.defaultdict(lambda: collections.defaultdict(list))
        
    # def _get_valid_field_value(self, timestamp: int, key: str, field: str) -> int | None:
    #     """
    #     Helper method to retrieve a field's value if it exists and has not expired
    #     """
    #     if key in self.records and field in self.records[key]:
    #         value, expiration_ts = self.records[key][field]
    #         if expiration_ts is None or timestamp < expiration_ts:
    #             return value
    #     return None
        
    def _get_latest_valid_value(self, timestamp: int, key: str, field: str) -> int | None:
        if key not in self.records or field not in self.records[key]:
            return None
        
        history = self.records[key][field]
        if not history:
            return None
        
        _, value, expiration_ts = history[-1]
        if value is DELETED:
            return None
            
        if expiration_ts is not None and timestamp >= expiration_ts:
            return None
            
        return value
        
    def _find_entry_at_timestamp(self, key: str, field: str, at_timestamp: int):
        if key not in self.records or field not in self.records[key]:
            return None
        
        history = self.records[key][field]
        timestamps = [entry[0] for entry in history]
        
        # Return the index where to insert item x in list a, assuming a is sorted.
        # we want the latest entry before at_timestamp
        # bisect_right loops from the right
        idx = bisect.bisect_right(a=timestamps, x=at_timestamp)
        
        if idx == 0:
            return None
        
        _, value, expiration_ts = history[idx - 1]
        
        if value is DELETED:
            return None
        
        if expiration_ts is not None and at_timestamp >= expiration_ts:
            return None
            
        return value
        

    # TODO: implement interface methods here
    def set(self, timestamp: int, key: str, field: str, value: int) -> None:
        # A 'None' expiration time represents an infinite TTL
        self.records[key][field].append((timestamp, value, None))
        
    def set_with_ttl(self, timestamp: int, key: str, field: str, value: int, ttl: int) -> None:
        # should insert the specified value and set its Time-To-Live starting at timestamp
        # if the field in the record already exists, then update its value and TTL
        # The ttl parameter represents the number of time units that this field-value pair should exist in the database, meaning it will be available during this interval: [timestamp, timestamp + ttl]. It is guaranteed that ttl is greater than 0
        expiration_ts = timestamp + ttl
        self.records[key][field].append((timestamp, value, expiration_ts))

    def get(self, timestamp: int, key: str, field: str) -> int | None:
        # should return the value contained within field of the record associated with key. 
        # If the record or the field does not exist, should return None
        return self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
        
    def get_when(self, timestamp: int, key: str, field: str, at_timestamp: int) -> int | None:
        # should return the value of field at `at_timestamp` from the record associated with key
        # if `at_timestamp` is 0, perform the get operation described in Level 1
        if at_timestamp == 0:
            return self.get(timestamp=timestamp, key=key, field=field)
        # It is guaranteed that at_timestamp will not be greater than `timestamp`
        # If the specified `field` or record did not exist at the given timestamp, return None
        return self._find_entry_at_timestamp(key=key, field=field, at_timestamp=at_timestamp)
        
    def compare_and_set(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int) -> bool:
        # should update the value of field in the record assocaited with key to new_value if the current value equals expected_value
        current_value = self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
        if current_value == expected_value:
            self.records[key][field].append((timestamp, new_value, None))
            return True
        # if expected_value does not match the current value, or either
        # key or field does not exist, this operation is ignored
        return False
        
    def compare_and_set_with_ttl(self, timestamp: int, key: str, field: str, expected_value: int, new_value: int, ttl: int) -> bool:
        # the same as compare_and_set, but should also update TTL of the new value
        current_value = self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
        if current_value == expected_value:
            expiration_ts = timestamp + ttl
            self.records[key][field].append((timestamp, new_value, expiration_ts))
            return True
        # This operation should return True if the field waas updated
        # and False otherwise
        return False
        
    def compare_and_delete(self, timestamp: int, key: str, field: str, expected_value: int) -> bool:
        # should remove the field in the record associated with key if the previous value equals expected_value. 
        current_value = self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
        if current_value == expected_value:
            if key in self.records and field in self.records[key]:
                # del self.records[key][field]
                self.records[key][field].append((timestamp, DELETED, None))
                return True
        # If expected_value does not match the current value,
        # or either key or field does not exist, the operation is ignored
        return False
        
    def scan(self, timestamp: int, key: str) -> list[str]:
        # If the specified record does not exist, returns an empty list
        if key not in self.records:
            return []
        # should return a list of strings representing the fields of the record associated with key. The returned list should be in the following format
        # ["<field_1>(<value_1>)", "<field_2>(<value_2>)", ...]
        # where fields are sorted lexicographically
        result = []
        for field in sorted(self.records[key].keys()):
            value = self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
            if value is not None:
                result.append(f"{field}({value})")
        return result
    
    def scan_by_prefix(self, timestamp: int, key: str, prefix: str) -> list[str]:
        # should return a list of strings representing some fields of the record associated with key
        # specifically, only fields that start with prefix should be included.
        # the returned list should be in the same format as in the scan operation with fields sorted in lexicographical order
        if key not in self.records:
            return []
        
        result = []
        matching_fields = sorted([f for f in self.records[key] if f.startswith(prefix)])
        
        for field in matching_fields:
            value = self._get_latest_valid_value(timestamp=timestamp, key=key, field=field)
            if value is not None:
                result.append(f"{field}({value})")
        return result
        
        
    
